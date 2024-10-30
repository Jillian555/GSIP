# For graph learning
import copy

import torch
from torch_geometric import seed_everything

# Utility
import os
import argparse
import sys
import numpy as np
from utilities import *
from data_stream import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_batch, eval_node_classifier


def evaluate(args, dataset, data_stream, memory_banks, flush=True):
    APs = []
    AFs = []
    mAPs = []
    Ps = []
    for i in range(args.repeat):
        memory_bank = memory_banks[i]
        # Initialize the performance matrix.
        performace_matrix = torch.zeros(len(memory_bank), len(memory_bank))
        model = get_backbone_model(dataset, data_stream, args)
        cgl_model = get_cgl_model(model, data_stream, args)
        tasks = cgl_model.tasks
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        mAP = 0
        prev_model = None
        for k in range(len(memory_bank)):
            # train
            if args.dataset_name == "products" and args.cgl_method == "joint":
                max_cls = torch.unique(memory_bank[k].y)[-1]
                batches = memory_bank[:k + 1]
                for data in batches:
                    data.to(args.device)
                model = train_node_classifier_batch(model, batches, opt, n_epoch=args.cls_epoch,
                                                    incremental_cls=(0, max_cls + 1))
            else:
                if args.tim:
                    if args.batch:
                        if args.sep == 1:
                            if k == 0:
                                replayed_graphs = [None, memory_bank[:k + 1]]
                            else:
                                replayed_graphs = [memory_bank[:k + 1], memory_bank[:k + 1]]
                        else:
                            replayed_graphs = memory_bank[:k + 1]
                    else:
                        if args.sep == 1:
                            if k == 0:
                                replayed_graphs = [None, Batch.from_data_list(memory_bank[:k + 1])]
                            else:
                                replayed_graphs = [Batch.from_data_list(memory_bank[:k + 1]),
                                                   Batch.from_data_list(memory_bank[:k + 1])]
                        else:
                            replayed_graphs = Batch.from_data_list(memory_bank[:k + 1])

                else:
                    if args.batch:
                        if args.sep == 1:
                            if k == 0:
                                replayed_graphs = [None, [tasks[k:k + 1]]]
                            else:
                                replayed_graphs = [memory_bank[:k + 1], memory_bank[:k + 1] + [tasks[k:k + 1]]]
                        else:
                            replayed_graphs = memory_bank[:k + 1] + [tasks[k:k + 1]]
                    else:
                        if args.sep == 1:
                            if k == 0:
                                replayed_graphs = [None, Batch.from_data_list([tasks[k:k + 1]])]
                            else:
                                replayed_graphs = [Batch.from_data_list(memory_bank[:k + 1]),
                                                   Batch.from_data_list(memory_bank[:k + 1] + [tasks[k:k + 1]])]
                        else:
                            replayed_graphs = Batch.from_data_list(memory_bank[:k + 1] + [tasks[k:k + 1]])

                if args.batch:
                    batches = replayed_graphs
                    max_cls = torch.unique(memory_bank[k].y)[-1]
                    if args.sep == 1:
                        if k == 0:
                            for data in batches[1]:
                                data.to(args.device)
                        else:
                            for data in batches[0]:
                                data.to(args.device)
                            for data in batches[1]:
                                data.to(args.device)
                    else:
                        for data in batches:
                            data.to(args.device)
                    model = train_node_classifier_batch(model, batches, opt, n_epoch=args.cls_epoch,
                                                                       incremental_cls=(0, max_cls + 1), args=args, t=k,
                                                                       prev_model=prev_model)
                    prev_model = copy.deepcopy(model)
                else:
                    if args.sep == 1:
                        if k == 0:
                            replayed_graphs = [None, replayed_graphs[1].to(args.device, "x", "y", "adj_t")]
                        else:
                            replayed_graphs = [replayed_graphs[0].to(args.device, "x", "y", "adj_t"),
                                               replayed_graphs[1].to(args.device, "x", "y", "adj_t")]
                        max_cls = torch.unique(replayed_graphs[1].y)[-1]
                    else:
                        replayed_graphs.to(args.device, "x", "y", "adj_t")
                        max_cls = torch.unique(replayed_graphs.y)[-1]
                    print('max_cls', max_cls)
                    model = train_node_classifier(model, replayed_graphs, opt, weight=None,
                                                                 n_epoch=args.cls_epoch,
                                                                 incremental_cls=(0, max_cls + 1), args=args, t=k,
                                                                 prev_model=prev_model)
                    prev_model = copy.deepcopy(model)

            # Test the model from task 0 to task k
            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(args.device, "x", "y", "adj_t")
                if args.IL == "classIL":
                    acc = eval_node_classifier(model, task_, incremental_cls=(0, max_cls + 1)) * 100
                else:
                    max_cls = torch.unique(task_.y)[-1]
                    acc = eval_node_classifier(model, task_, incremental_cls=(
                    max_cls + 1 - data_stream.cls_per_task, max_cls + 1)) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=flush)
                performace_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            mAP += AP
            print(f"AP: {AP:.2f}", end=", ", flush=flush)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=flush)
        APs.append(AP)
        AFs.append(AF)
        mAPs.append(mAP / (k + 1))
        Ps.append(performace_matrix)
    print(f"AP: {np.mean(APs):.2f}±{np.std(APs, ddof=1):.2f}", flush=flush)
    # print(f"mAP: {np.mean(mAPs):.1f}±{np.std(mAPs, ddof=1):.2f}", flush=flush)
    print(f"AF: {np.mean(AFs):.2f}±{np.std(AFs, ddof=1):.2f}", flush=flush)
    return Ps


def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--data-dir', type=str, default="/root/code/cat/data")
    parser.add_argument('--result-path', type=str, default="/root/code/cat/results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', type=bool, default=True)
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200)
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--m-update', type=str, default="all")
    parser.add_argument('--cgm-args', type=str,
                        default="{'n_encoders': 500, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}")
    parser.add_argument('--IL', type=str, default="classIL")
    parser.add_argument('--batch', action='store_true')

    parser.add_argument('--n_base', type=int, default=20, help='how many base classes')
    parser.add_argument('--n_cls_per_task', type=int, default=30, help='how many classes does each task contain')
    parser.add_argument('--n_task', type=int, default=3, help='how many tasks')
    parser.add_argument('--sep', type=int, default=1, help='distinguish between new and old classes')
    parser.add_argument('--neibt', type=float, default=0.99, help='threshold for neighbor selection')
    parser.add_argument('--neibt1', type=float, default=0.9, help='threshold for neighbor selection')
    parser.add_argument('--w_ll', type=float, default=0.0, help='loss weight of low-frequency local information preservation')
    parser.add_argument('--w_lg', type=float, default=0.0, help='loss weight of low-frequency global information preservation')
    parser.add_argument('--w_h', type=float, default=0.0, help='loss weight of high-frequency information preservation')

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--rewrite', action='store_true')

    args = parser.parse_args()
    print('-----------')
    print(args)

    seed_everything(args.seed)

    # Get file names.
    result_file_name = get_result_file_name(args)
    memory_bank_file_name = os.path.join(args.result_path, "memory_bank", result_file_name)
    task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")

    dataset = get_dataset(args)
    if os.path.exists(task_file):
        data_stream = torch.load(task_file)
    else:
        data_stream = Streaming(args, dataset)
        torch.save(data_stream, task_file)


    # Get memory banks.
    memory_banks = []
    for i in range(args.repeat):
        if os.path.exists(memory_bank_file_name + f"_repeat_{i}") and not args.rewrite:
            memory_bank = torch.load(memory_bank_file_name + f"_repeat_{i}")
            memory_banks.append(memory_bank)  # load the memory bank from the file.
        else:
            model = get_backbone_model(dataset, data_stream, args)
            cgl_model = get_cgl_model(model, data_stream, args)

            memory_bank = cgl_model.observer()
            memory_banks.append(memory_bank)
            torch.save(memory_bank, memory_bank_file_name + f"_repeat_{i}")

    Ps = evaluate(args, dataset, data_stream, memory_banks)

    if args.tim:
        if args.batch:
            torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_tim_batch.pt"))
        else:
            torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_tim.pt"))
            import pickle
            with open(os.path.join(args.result_path, "performance_new",
                                   f"{args.dataset_name}_{args.n_base}_{args.n_cls_per_task}_{args.cgl_method}.pkl"),
                      'wb') as f:
                pickle.dump(Ps, f)
    else:
        if args.batch:
            torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_batch.pt"))
        else:
            torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}.pt"))


if __name__ == '__main__':
    main()