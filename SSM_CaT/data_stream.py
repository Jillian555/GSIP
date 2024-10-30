import torch
import random

from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor
from progressbar import progressbar



class Streaming():
    def __init__(self, args, dataset):
        self.tasks = self.prepare_tasks(args, dataset)
    def prepare_tasks(self, args, dataset):
        graph = dataset[0]

        tasks = []
        cls = [[i for i in range(args.n_base)]]
        for i in range(args.n_base, args.n_cls - 1, args.n_cls_per_task):
            cls.append(list(range(i, i + args.n_cls_per_task)))

        k=0
        for classes in cls:
            subset = sum(graph.y == cls for cls in classes).squeeze().nonzero(as_tuple=False)
            subgraph = graph.subgraph(subset.squeeze())
            
            # Split to train/val/test
            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric()  # Arxiv is an directed graph.

            subgraph.task_id = k
            k+=1
            subgraph.classes = classes

            tasks.append(subgraph)
        return tasks
