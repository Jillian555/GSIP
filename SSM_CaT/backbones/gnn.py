import torch
import torch.nn.functional as F


# from focal_loss.focal_loss import FocalLoss


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

    def initialize(self):
        for layer in self.layers:
            # torch.nn.init.normal_(layer.lin.weight.data)
            layer.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
        x = self.layers[-1](x, adj_t)
        return x
        # return F.log_softmax(x, dim=1)

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
            x = F.relu(x)
        return x

    def encode_noise(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_t)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        random_noise = torch.rand_like(x).cuda()
        x += torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        return x




def train_node_classifier(model, data, optimizer, weight=None, n_epoch=200, incremental_cls=None, args=None, t=None,
                          prev_model=None):
    if args.sep == 1:
        old_data, new_data = data[0], data[1]
        data = new_data
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    mse = torch.nn.MSELoss()
    loss_ll = loss_lg = loss_h = loss_m = 0
    w_m = 0
    for epoch in range(n_epoch):
        print(epoch, '-----------------------------------------------------')
        out = model(data)
        loss = ce(out[data.train_mask, 0:incremental_cls[1]], data.y[data.train_mask])
        d = data.adj_t.t().to_dense().to(args.device)
        if t > 0:
            if args.sep == 1:
                out = model(old_data)
            output_old = prev_model(data)
            output_new = out[data.train_mask, :]
            output_old = output_old[data.train_mask, :]
            if args.cgl_method == 'ssm':
                d += torch.eye(d.shape[0]).to(args.device)
                neibors = torch.nonzero(d).squeeze().t()
                neibors2 = torch.nonzero(torch.matmul(d, d)).squeeze().t()
            elif args.cgl_method == 'cgm':
                similarity_fea = F.cosine_similarity(output_old.unsqueeze(1), output_old.unsqueeze(0),
                                                     dim=2) > args.neibt
                similarity_fea1 = F.cosine_similarity(output_old.unsqueeze(1), output_old.unsqueeze(0),
                                                      dim=2) > args.neibt1
                similarity_lab = torch.eq(old_data.y.unsqueeze(dim=-1), old_data.y.unsqueeze(dim=-1).t())
                similarity = similarity_fea * similarity_lab
                neibors = torch.nonzero(similarity).squeeze().t()
                similarity1 = similarity_fea1
                neibors1 = torch.nonzero(similarity1).squeeze().t()
            else:
                pass
            if args.w_ll != 0:
                output_new_ = output_new[neibors[0]]
                output_old_ = output_old[neibors[1]]
                loss_ll = mse(output_new_, output_old_.detach())
                print('loss_ll {:.4f}'.format(loss_ll.item()))
            if args.w_lg != 0:
                loss_lg = mse(output_new.mean(dim=0), output_old.mean(dim=0).detach())
                print('loss_lg {:.4f}'.format(loss_lg.item()))
            if args.w_h != 0:
                if args.cgl_method == 'ssm':
                   neibors1 = neibors
                embeddings_ = torch.abs(output_new[neibors1[0]] - output_new[neibors1[1]])
                embeddings_old = torch.abs(output_old[neibors1[0]] - output_old[neibors1[1]])
                loss_h = F.kl_div(F.log_softmax(embeddings_ / 1.0, dim=-1),
                                     F.softmax(embeddings_old.detach() / 1.0, dim=-1), reduction='batchmean')
                print('loss_h {:.4f}'.format(loss_h.item()))
            if args.cgl_method == 'ssm' and w_m != 0:
                embeddings_hop = torch.abs(output_new[neibors2[0]] - output_new[neibors2[1]])
                embeddings_hop_old = torch.abs(output_old[neibors2[0]] - output_old[neibors2[1]])
                loss_m = F.kl_div(F.log_softmax(embeddings_hop / 1.0, dim=-1),
                                    F.softmax(embeddings_hop_old.detach() / 1.0, dim=-1), reduction='batchmean')
                print('loss_m {:.4f}'.format(loss_m.item()))
        loss = loss + args.w_ll * loss_ll + args.w_lg * loss_lg + args.w_h * loss_h + w_m * loss_m
        print('loss {:.4f}'.format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_node_classifier_batch(model, batches, optimizer, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(n_epoch):
        for data in batches:
            if incremental_cls:
                out = model(data)[:, 0:incremental_cls[1]]
            else:
                out = model(data)

            loss = ce(out[data.train_mask], data.y[data.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    pred = model(data)[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
    correct = (pred == data.y[data.test_mask] - incremental_cls[0]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
