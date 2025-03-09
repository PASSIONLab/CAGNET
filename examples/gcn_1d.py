import argparse
from collections import defaultdict
from itertools import accumulate
import copy
import math
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
import cagnet.nn.functional as CAGF

import socket

class GCN(nn.Module):
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, rank, size, timers, partitioning, sparse_unaware,
                    device, group=None, row_groups=None, col_groups=None):
    super(GCN, self).__init__()
    self.layers = nn.ModuleList()
    self.rank = rank
    self.size = size
    self.group = group
    self.sparse_unaware = sparse_unaware
    self.device = device
    self.timers = timers
    self.partitioning = partitioning
    self.timings = defaultdict(float)

    # input layer
    self.layers.append(GCNConv(in_feats, n_hidden, self.partitioning, self.device))
    # hidden layers
    for i in range(n_layers - 1):
        self.layers.append(GCNConv(n_hidden, n_hidden, self.partitioning, self.device))
    # output layer
    self.layers.append(GCNConv(n_hidden, n_classes, self.partitioning, self.device))

  def forward(self, graph, inputs, ampbyp, epoch):
    h = inputs
    self.epoch = epoch
    for l, layer in enumerate(self.layers):
      h = layer(self, graph, h, ampbyp)
      if l != len(self.layers) - 1:
        h = CAGF.relu(h, self.partitioning)

    h = CAGF.log_softmax(self, h, self.partitioning, dim=1)
    return h

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx, normalization):
    if not normalization:
        return adj_part

    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].double(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))

    return adj_part

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, partitions, dim):

    # vtx_indices = list(range(0, node_count, n_per_proc))
    # vtx_indices.append(node_count)

    vtx_indices = [0]
    vtx_indices.extend(list(accumulate(partitions)))

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def oned_partition(rank, size, inputs, adj_matrix, data, features, classes, device, normalize, partitions=[]):
    node_count = inputs.size(0)

    if not partitions:
        n_per_proc = math.ceil(float(node_count) / size)
        partitions = [math.ceil(float(node_count) / size)]*size
        partitions[size-1] = inputs.size(0) - math.ceil(float(node_count) / size)*(size - 1)    

    am_partitions = None
    am_pbyp = None

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, partitions, 1)

        proc_node_count = vtx_indices[rank + 1] - vtx_indices[rank]
        am_pbyp, _ = split_coo(am_partitions[rank], partitions, 0)
        for i in range(len(am_pbyp)):
            if i == size - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank], normalize)
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(vtx_indices[i + 1] - vtx_indices[i], proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank], normalize)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i], normalize)

        input_partitions = torch.split(inputs, partitions, dim=0)

        adj_matrix_loc = am_partitions[rank]
        inputs_loc = input_partitions[rank]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs.size: {inputs.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def evaluate(model, graph, features, labels, nid, ampbyp):
    model.eval()
    with torch.no_grad():
        non_eval_timings = copy.deepcopy(model.timings)
        logits = model(graph, features, ampbyp, 0) # Pass 0 in for epoch to avoid timing
        model.timings = non_eval_timings # don't include evaluation timings

        # # all-gather logits across ranks
        # logits_recv = []
        # for i in range(len(ampbyp)):
        #     logits_recv.append(torch.FloatTensor(ampbyp[0].size(1), logits.size(1)))

        # if logits.size(0) != ampbyp[0].size(1):
        #     pad_row = ampbyp[0].size(1) - logits.size(0)
        #     logits = torch.cat((logits, torch.FloatTensor(pad_row, logits.size(1))), dim=0)

        # dist.all_gather(logits_recv, logits, group)

        # padding = graph.size(0) - ampbyp[0].size(1) * (len(ampbyp) - 1)
        # logits_recv[-1] = logits_recv[-1][:padding,:]

        # logits = torch.cat(logits_recv)

        logits = logits[nid]
        labels = labels[nid]
        if logits.size(0) > 0:
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
        else:
            correct = torch.tensor(0)

        dist.all_reduce(correct, op=dist.reduce_op.SUM)
        return correct.item() * 1.0 / nid.size(0)
        # return correct.item() * 1.0 / len(labels)

def main(args):
    # Initialize distributed environment with SLURM
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"
    
    dist.init_process_group(backend=args.dist_backend)
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}", flush=True)
    print(rank % args.gpu)
    torch.cuda.set_device(rank % args.gpu)

    # load and preprocess dataset
    device = torch.device(f'cuda:{rank % args.gpu}')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)

    if args.dataset == "Cora" or args.dataset == "Reddit":
        if args.dataset == "Cora":
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset == "Reddit":
            dataset = Reddit(path)

        data = dataset[0]
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        adj_matrix = edge_index
    elif args.dataset == "Amazon":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../../data/Amazon/processed/data.pt")
        print(f"Done loading coo", flush=True)
        n = 9430088
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Small_16":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../../data/Amazon_Small_16/processed/Amazon_Small_16.pt")
        print(f"Done loading coo", flush=True)
        # n = 14249639
        n = 9430088
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Small_64":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../../data/Amazon_Small_64/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Small_16_graph_vb":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../../data/Amazon_Small_16_graph-vb/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 9430088
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_4":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_4_graph_vb/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_8":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_8_graph_vb/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_4M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_4M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_16":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_16_graph_vb/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_16M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_16M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_32":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_32_graph_vb/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_32M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_32M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_64":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_64_graph_vb/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Reddit_64M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Reddit_64M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 232965
        num_features = 602
        num_classes = 41
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_4":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_4/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_4":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_4M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_4M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_4M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_16":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_16/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_16M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_16M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_32":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_32/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_32M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_32/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_64":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_64/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_64M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_64M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_128":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_128/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Amazon_Large_256":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_256/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

    elif args.dataset == "Protein_4":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_4/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

    elif args.dataset == "Protein_4M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_4M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

    elif args.dataset == "Protein_16":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_16/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_16M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_16M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

    elif args.dataset == "Protein_32":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_32/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_32M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_32M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_64":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_64/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_64M":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_64M/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_128":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_128/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_256":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_256/processed/amazon_large_randomized.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542   
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Papers":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/global/cfs/cdirs/m1982/alokt/data/ogbn_papers100M/processed/papers_sym.pt")
        print(f"Done loading coo", flush=True)
        n = 111059956 
        num_features = 100
        num_classes = 172
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    

    if args.normalize:
        adj_matrix, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))

    partitioning = Partitioning.ONED
    partitions = []

    if args.partitions:
        partitions_file = open(args.partitions, "r")
        partitions_data = partitions_file.read().strip()
        print(partitions_data)
        partitions = [int(x) for x in partitions_data.split("\n")]

    print(f"partitioning...", flush=True)
    group = dist.new_group(list(range(size)))
    features_loc, g_loc, ampbyp = oned_partition(rank, size, inputs, adj_matrix, data, \
                                                      inputs, num_classes, device, args.normalize, partitions)
    print(f"done partitioning", flush=True)

    features_loc = features_loc.to(device)
    g_loc = g_loc.to(device)
    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)


    # create GCN model
    torch.manual_seed(0)
    model = GCN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.n_layers,
                      rank,
                      size,
                      args.timers,
                      partitioning,
                      args.sparse_unaware,
                      device,
                      group)

    counts_send = []
    row_indices_send = []

    counts_recv = [torch.cuda.LongTensor(1, 1, device=device).fill_(0) for i in range(size)]

    for i in range(size):
        unique_cols = ampbyp[i]._indices()[1].unique()
        counts_send.append(torch.cuda.LongTensor([unique_cols.size()], device=device).resize_(1, 1))
        row_indices_send.append(unique_cols)
    
    model.row_indices_send = row_indices_send

#    print("all to all counts")  
    dist.all_to_all(counts_recv, counts_send)
 
    row_indices_recv = [torch.cuda.LongTensor(device=device).resize_(counts_recv[i].int().item(),).fill_(0) for i in range(len(counts_recv))]

    dist.all_to_all(row_indices_recv, row_indices_send, group=group)

    model.row_indices_recv = row_indices_recv

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_per_proc = math.ceil(float(inputs.size(0)) / size)

    rank_train_mask = []
    labels_rank = []

    if partitions:
        rank_train_mask = torch.split(data.train_mask, partitions, dim=0)[rank]
        labels_rank = torch.split(data.y, partitions, dim=0)[rank]

        rank_val_mask = torch.split(data.val_mask, partitions, dim=0)[rank]
        labels_rank = torch.split(data.y, partitions, dim=0)[rank]
    else:
        rank_train_mask = torch.split(data.train_mask, n_per_proc, dim=0)[rank]
        labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank]

        rank_val_mask = torch.split(data.val_mask, n_per_proc, dim=0)[rank]
        labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank]

    rank_train_nids = rank_train_mask.nonzero().squeeze()
    rank_val_nids = rank_val_mask.nonzero().squeeze()

    train_nid = data.train_mask.nonzero().squeeze()

    torch.manual_seed(0)
    dur = []

    for epoch in range(args.n_epochs):
        if epoch >= 1:
            epoch_start = time.time()
        model.train()

        # forward
        logits = model(g_loc, features_loc, ampbyp, epoch)
        loss = CAGF.cross_entropy(logits[rank_train_nids], labels_rank[rank_train_nids], train_nid.size(0), \
                                        num_classes, partitioning, rank, group, size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if epoch >= 1:
            dur.append(time.time() - epoch_start)
        acc = evaluate(model, g_loc, features_loc, labels_rank, rank_val_nids, ampbyp)
        print("Rank: {:05d} | Epoch {:05d} | Epoch Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}"\
                        .format(rank, epoch, np.mean(dur), loss.item(), acc))
    dist.barrier()
    print(f"rank: {rank} Total Time: {np.sum(dur)}", flush=True)
    print(f"rank: {rank} timings: {model.timings}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=4,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument('--normalize', action="store_true",
                            help='normalize adjacency matrix')
    parser.add_argument('--partitioning', default='ONED', type=str,
                            help='partitioning strategy to use')
    parser.add_argument('--timers', action="store_true",
                            help='turn on timers')
    parser.add_argument('--partitions', default='', type=str,
                            help='text file for unequal partitions')
    parser.add_argument('--sparse-unaware', action="store_true",
                            help='use sparsity-unaware implementation')
    args = parser.parse_args()
    print(args)

    main(args)
