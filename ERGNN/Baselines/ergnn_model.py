import torch
from .ergnn_utils import *
import torch.nn.functional as F

samplers = {'CM': CM_sampler(plus=False), 'CM_plus': CM_sampler(plus=True), 'MF': MF_sampler(plus=False),
            'MF_plus': MF_sampler(plus=True), 'random': random_sampler(plus=False)}
class NET(torch.nn.Module):
    """
        ER-GNN baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """

    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model
        self.sampler = samplers[args.ergnn_args['sampler']]

        # setup losses
        self.ce = torch.nn.functional.cross_entropy
        self.mse = torch.nn.MSELoss()

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = int(args.ergnn_args['budget'])
        self.d_CM = args.ergnn_args['d']  # d for CM sampler of ERGNN
        self.aux_g = None


    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks under the class-IL setting.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class.
                :param dataset: The entire dataset.

                """

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        beta = buffer_size / (buffer_size + n_nodes)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], labels[train_ids], weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], labels[train_ids], weight=loss_w_)

        if t != self.current_task:
            # if the incoming task is new
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h, self.d_CM)
            old_ids = g.ndata['_ID'].cpu()  # '_ID' are the original ids in the original graph before splitting
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            if t > 0:
                g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                self.aux_g = g.to(device='cuda:{}'.format(features.get_device()))
                self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        if t != 0:
            # calculate auxiliary loss based on replay if not the first task
            output, _ = self.net(self.aux_g, self.aux_features)
            if args.classifier_increase:
                loss_aux = self.ce(output[:, offset1:offset2], self.aux_labels,
                                   weight=self.aux_loss_w_[offset1: offset2])
            else:
                loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)
            loss = beta * loss + (1 - beta) * loss_aux

        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class.
                        :param dataset: The entire dataset.

                        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = 0
        for k in self.buffer_node_ids:
            buffer_size += len(self.buffer_node_ids[k])
        beta = buffer_size / (buffer_size + n_nodes)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels - offset1, weight=loss_w_[offset1: offset2])

        if t != self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h, self.d_CM)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[t])
            self.aux_g.append(g.to(device='cuda:{}'.format(features.get_device())))
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            self.aux_loss_w_.append(loss_w_)

        if t != 0:
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1: o2])
                loss = beta * loss + (1 - beta) * loss_aux

        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        self.net.train()
        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = 0
            for k in self.buffer_node_ids:
                buffer_size += len(self.buffer_node_ids[k])
            beta = buffer_size / (buffer_size + n_nodes_current_batch)
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_labels = output_labels - offset1
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

            # sample and store ids from current task
            if t != self.current_task:
                self.current_task = t
                sampled_ids = self.sampler(ids_per_cls_train, self.budget,
                                           features.to(device='cuda:{}'.format(args.gpu)), self.net.second_last_h,
                                           self.d_CM)
                old_ids = g.ndata['_ID'].cpu()
                self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
                ag, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[t])
                self.aux_g.append(ag.to(device='cuda:{}'.format(args.gpu)))
                if args.cls_balance:
                    n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)

            if t != 0:
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[
                        1]
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1:o2])
                    loss = beta * loss + (1 - beta) * loss_aux
            loss.backward()
            self.opt.step()

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids,
                               ids_per_cls, dataset):
        """
                                The method for learning the given tasks under the class-IL setting with mini-batch training.

                                :param args: Same as the args in __init__().
                                :param g: The graph of the current task.
                                :param dataloader: The data loader for mini-batch training
                                :param features: Node features of the current task.
                                :param labels: Labels of the nodes in the current task.
                                :param t: Index of the current task.
                                :param train_ids: The indices of the nodes participating in the training.
                                :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                                :param dataset: The entire dataset (currently not in use).

                                """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = len(self.buffer_node_ids)
            beta = buffer_size / (buffer_size + n_nodes_current_batch)
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions, _ = self.net.forward_batch(blocks, input_features)

            if output_labels.shape == torch.Size([]):
                output_labels = output_labels.unsqueeze(dim=0)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

            # sample and store ids from current task
            if t != self.current_task:
                self.current_task = t
                old_ids = g.ndata['_ID'].cpu()
                sampled_ids = self.sampler(ids_per_cls_train, self.budget,
                                           features.to(device='cuda:{}'.format(args.gpu)), self.net.second_last_h,
                                           self.d_CM, using_half=False)
                self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())

                if t > 0:
                    self.aug_gs = []
                    batch_size = 1000
                    for i in range(0, len(self.buffer_node_ids), batch_size):
                        if (i + batch_size) < len(self.buffer_node_ids):
                            aux_g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[i:i + batch_size])
                        else:
                            aux_g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[i:len(self.buffer_node_ids)])
                        self.aux_g = aux_g.to(device='cuda:{}'.format(args.gpu))
                        self.aug_gs.append(self.aux_g)

            if t > 0:
                loss_aux, loss_ll, loss_lg, loss_h = 0, 0, 0, 0
                for iii in range(len(self.aug_gs)):
                    self.aux_g = self.aug_gs[iii]
                    self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata[
                        'label'].squeeze()
                    if args.cls_balance:
                        n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                        loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                    else:
                        loss_w_ = [1. for i in range(args.n_cls)]
                    self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                    output, _ = self.net(self.aux_g, self.aux_features)
                    num = len(output)
                    if self.aux_labels.shape == torch.Size([]):
                        self.aux_labels = self.aux_labels.unsqueeze(dim=0)
                    if args.classifier_increase:
                        loss_aux_ = self.ce(output[:, offset1:offset2], self.aux_labels,
                                           weight=self.aux_loss_w_[offset1: offset2])
                    else:
                        loss_aux_ = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)
                    loss_aux += loss_aux_ * num

                    output_new = output
                    output_old, _ = prev_model(self.aux_g, self.aux_features)
                    similarity_fea = F.cosine_similarity(output_old.unsqueeze(1), output_old.unsqueeze(0),
                                                         dim=2) > args.neibt
                    similarity_fea1 = F.cosine_similarity(output_old.unsqueeze(1), output_old.unsqueeze(0),
                                                          dim=2) > args.neibt1
                    similarity_lab = torch.eq(self.aux_labels.unsqueeze(dim=-1), self.aux_labels.unsqueeze(dim=-1).t())
                    similarity = similarity_fea * similarity_lab
                    similarity1 = similarity_fea1
                    neibors = torch.nonzero(similarity).squeeze().t()
                    neibors1 = torch.nonzero(similarity1).squeeze().t()

                    if args.ergnn_args['w_ll'] != 0:
                        output_new_ = output_new[neibors[0]]
                        output_old_ = output_old[neibors[1]]
                        loss_ll_ = self.mse(output_new_, output_old_.detach())
                        loss_ll += loss_ll_ * num
                        print('loss_ll_ {:.4f}'.format(loss_ll_.item()), output_new_.shape)
                    if args.ergnn_args['w_lg'] != 0:
                        loss_lg_ = self.mse(output_new.mean(dim=0), output_old.mean(dim=0).detach())
                        loss_lg += loss_lg_ * num
                        print('loss_lg_ {:.4f}'.format(loss_lg_.item()))
                    if args.ergnn_args['w_h'] != 0:
                        embeddings_ = torch.abs(output_new[neibors1[0]] - output_new[neibors1[1]])
                        embeddings_old = torch.abs(output_old[neibors1[0]] - output_old[neibors1[1]])
                        loss_h_ = F.kl_div(F.log_softmax(embeddings_ / 1.0, dim=-1),
                                             F.softmax(embeddings_old.detach() / 1.0, dim=-1), reduction='batchmean')
                        print('loss_h_ {:.4f}'.format(loss_h_.item()))
                        loss_h += loss_h_ * num
                total_num = len(self.buffer_node_ids)
                loss = beta * loss + (1 - beta) * loss_aux / total_num + args.ergnn_args['w_ll'] * loss_ll / total_num \
                    + args.ergnn_args['w_lg'] * loss_lg / total_num + args.ergnn_args['w_h'] * loss_h / total_num
            print('loss {:.4f}'.format(loss.item()))
            loss.backward()
            self.opt.step()
            torch.cuda.empty_cache()
