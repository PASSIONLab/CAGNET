import argparse
from collections import defaultdict
from itertools import accumulate
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
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, rank, size, timers, partitioning, replication, device,
                    group=None, row_groups=None, col_groups=None):
    super(GCN, self).__init__()
    self.layers = nn.ModuleList()
    self.rank = rank
    self.size = size
    self.timers = timers
    self.group = group
    self.row_groups = row_groups
    self.col_groups = col_groups
    self.device = device
    self.partitioning = partitioning
    self.replication = replication
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

def get_proc_groups(rank, size, replication):
    rank_c = rank // replication
     
    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

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

def one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, normalize, device, partitions):
    node_count = inputs.size(0)
    
    if not partitions:
        n_per_proc = math.ceil(float(node_count) / (size // replication))
        partitions = [n_per_proc]*(size // replication)
        partitions[size//replication -1] = inputs.size(0) - math.ceil(float(node_count) / (size//replication))*((size//replication) - 1)  

    am_partitions = None
    am_pbyp = None

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, partitions, 1)
        print(vtx_indices)
        print(rank_c)
        proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        am_pbyp, _ = split_coo(am_partitions[rank_c], partitions, 0)
        for i in range(len(am_pbyp)):
            if i == size // replication - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank_c], normalize)
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(vtx_indices[i + 1] - vtx_indices[i], proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank_c], normalize)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i], \
                                                    normalize)

        input_partitions = torch.split(inputs, partitions, dim=0)

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    # return inputs_loc, adj_matrix_loc, am_pbyp
    return input_partitions, am_partitions, am_pbyp

def split_am_partition(am_partition, partitions, rank, size, replication, normalize):
    node_count = am_partition.size(0)
    if not partitions:
        n_per_proc = math.ceil(float(node_count) / (size // replication))
        partitions = [n_per_proc]*(size // replication)
        partitions[size//replication -1] = node_count - math.ceil(float(node_count) / (size//replication))*((size//replication) - 1)  

    am_pbyp = None
    print(am_partition.size())
    proc_node_count = am_partition.size(1) # column count
    proc_node_count_row = int(math.ceil(node_count / (size // replication)))
    am_pbyp, vtx_indices = split_coo(am_partition._indices(), partitions, 0)
    rank_c = size // replication
    rank_col = size % replication
    for i in range(len(am_pbyp)):
        if i == size // replication - 1:
            last_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                    size=(last_node_count, proc_node_count),
                                                    requires_grad=False)

            am_pbyp[i] = scale_elements(am_partition._indices(), am_pbyp[i], node_count, vtx_indices[i], 
                                            vtx_indices[rank_c], normalize)
        else:
            proc_node_count_row = vtx_indices[i + 1] - vtx_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                    size=(proc_node_count_row, proc_node_count),
                                                    requires_grad=False)

            am_pbyp[i] = scale_elements(am_partition._indices(), am_pbyp[i], node_count, vtx_indices[i], 
                                            vtx_indices[rank_c], normalize)

    return am_pbyp

def evaluate(model, graph, features, labels, nid, nid_count, ampbyp, group):
    model.eval()
    with torch.no_grad():
        non_eval_timings = copy.deepcopy(model.timings)
        logits = model(graph, features, ampbyp, degrees)
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

        dist.all_reduce(correct, op=dist.reduce_op.SUM, group=group)
        return correct.item() * 1.0 / nid_count
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
    torch.cuda.set_device(rank % args.gpu)

    # load and preprocess dataset
    device = torch.device(f'cuda:{rank % args.gpu}')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)

    if args.dataset == "Cora" or args.dataset == "Reddit":
        if args.dataset == "Cora":
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset == "Reddit":
            dataset = Reddit(path, transform=T.NormalizeFeatures())

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

    elif args.dataset == "Amazon_Large_4_graph_vb":
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

 

    elif args.dataset == "Amazon_Large_8":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Amazon_Large_8/processed/amazon_large_randomized.pt")
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

    elif args.dataset.startswith("ogb"):
        import ogb
        data = Data()
        from ogb.nodeproppred import PygNodePropPredDataset # only import if necessary, takes a long time
        if ("papers100M" in args.dataset and rank % args.gpu == 0) or "products" in args.dataset:
            dataset = PygNodePropPredDataset(name=args.dataset, root="../../data")
            data = dataset[0]

            split_idx = dataset.get_idx_split() 
            # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
            # valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
            # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
            # data = data.to(device)
            # data.x.requires_grad = True
            # inputs = data.x.to(device)
            inputs = data.x
            data.y = data.y.squeeze().to(device)
            print(f"data.y: {data.y}", flush=True)
            print(f"data.y.size: {data.y.size()}", flush=True)
            print(f"data.y.dtype: {data.y.dtype}", flush=True)
            # inputs.requires_grad = True
            # data.y = data.y.to(device)
            num_features = dataset.num_features
            edge_index = data.edge_index
            num_classes = dataset.num_classes
            adj_matrix = edge_index
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train'].to(device)
            test_idx = split_idx['test'].to(device)
    
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

    elif args.dataset == "Protein_8":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/pscratch/sd/j/jinimukh/Protein_8/processed/amazon_large_randomized.pt")
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

    elif args.dataset == "ogbn-papers100M":
        if rank % args.gpu == 0: 
            print(f"Loading coo...", flush=True)
            if args.partitions:
                # edge_index = torch.load("/pscratch/sd/r/roguzsel/zzz/sagnn/data/papers/base/papers_sym.pt")
                path = f"/global/cfs/cdirs/m1982/alokt/data/ogbn_papers100M/processed/papers_k{size}m1u1c10r2.pt"
                edge_index = torch.load(path)
            else:
                edge_index = torch.load("/global/cfs/cdirs/m1982/alokt/data/ogbn_papers100M/processed/papers_sym.pt")
            print(f"Done loading coo", flush=True)
            n = 111059956 
            num_features = 128
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

    partitioning = Partitioning.ONE5D

    partitions = []

    if args.partitions:
        partitions_file = open(args.partitions, "r")
        partitions_data = partitions_file.read().strip()
        # print(partitions_data)
        partitions = [int(x) for x in partitions_data.split("\n")]

    row_groups, col_groups = get_proc_groups(rank, size, args.replication)

    proc_row = size // args.replication
    rank_c = rank // args.replication
    rank_col = rank % args.replication
    if rank_c >= (size // args.replication):
        return

    if args.dataset == "ogbn-papers100M":
        if rank % args.gpu == 0:
            inputs = inputs.to(torch.device("cpu"))
            adj_matrix = adj_matrix.to(torch.device("cpu"))
            # send g_loc.nnz, g_loc, train_idx.len, and train_idx
            input_partitions, am_partitions, ampbyp = \
                    one5d_partition(rank, size, inputs, adj_matrix, data, 
                                        inputs, num_classes, args.replication, \
                                        args.normalize, device, partitions)
            features_loc = input_partitions[rank_c]
            g_loc = am_partitions[rank_c]

            for i in range(1, args.gpu):
                print(f"iteration i: {i}", flush=True)
                rank_send_row = (rank + i) // args.replication
                dst_gpu = rank + i
                g_send = am_partitions[rank_send_row]
                print("coalescing", flush=True)
                g_send = g_send.coalesce()
                print("normalizing", flush=True)
                g_send = g_send.to(device)
                g_send = g_send.double()
                features_send = input_partitions[rank_send_row].to(device)
                edge_count = adj_matrix.size(1)


                g_send_meta = torch.cuda.LongTensor(6)
                g_send_meta[0] = g_send._nnz()
                g_send_meta[1] = g_send.size(0)
                g_send_meta[2] = g_send.size(1)
                g_send_meta[3] = adj_matrix.size(1)
                g_send_meta[4] = num_features
                g_send_meta[5] = num_classes
                dist.send(g_send_meta, dst=dst_gpu)
                dist.send(g_send._indices(), dst=dst_gpu)
                dist.send(g_send._values(), dst=dst_gpu)
                dist.send(features_send, dst=dst_gpu)
                dist.send(data.y.float(), dst=dst_gpu)

                if args.dataset == "ogbn-papers100M":
                    train_idx_len = torch.cuda.LongTensor(1).fill_(train_idx.size(0))
                    test_idx_len = torch.cuda.LongTensor(1).fill_(test_idx.size(0))
                    dist.send(train_idx_len, dst=dst_gpu)
                    dist.send(test_idx_len, dst=dst_gpu)
                    dist.send(train_idx, dst=dst_gpu)
                    dist.send(test_idx, dst=dst_gpu)

                del g_send
                del features_send

            # features_loc, g_loc, _, _ = one5d_partition(rank, size, inputs, adj_matrix, data, inputs, num_classes, \
            #                                 args.replication, args.normalize)
            print("coalescing", flush=True)
            vtx_count = g_loc.size(0)
            g_loc = g_loc.coalesce().t_()
            print("normalizing", flush=True)
            g_loc = g_loc.to(device)
            features_loc = features_loc.to(device)
            g_loc = g_loc.double()
            edge_count = adj_matrix.size(1)
        else:
            src_gpu = rank - (rank % args.gpu)
            g_loc_meta = torch.cuda.LongTensor(6).fill_(0)
            dist.recv(g_loc_meta, src=src_gpu)

            g_loc_indices = torch.cuda.LongTensor(2, g_loc_meta[0].item())
            dist.recv(g_loc_indices, src=src_gpu)

            g_loc_values = torch.cuda.DoubleTensor(g_loc_meta[0].item())
            dist.recv(g_loc_values, src=src_gpu)

            num_features = g_loc_meta[4].item()
            num_classes = g_loc_meta[5].item()
            if rank_c < proc_row - 1:
                inputs = torch.cuda.FloatTensor(g_loc_meta[2].item(), num_features)
            else:
                if args.partitions:
                    rows = partitions[-1]
                else:
                    # rows = g_loc_meta[2].item() + (g_loc_meta[1].item() - proc_row * g_loc_meta[2].item())
                    normal_node_count = int(math.ceil(g_loc_meta[1].item() / (size // args.replication)))
                    rows = g_loc_meta[1].item() - (proc_row - 1) * normal_node_count
                print(f"rows: {rows}", flush=True)
                inputs = torch.cuda.FloatTensor(rows, num_features)
            dist.recv(inputs, src=src_gpu)

            # data = Data()
            data.y = torch.cuda.FloatTensor(g_loc_meta[1].item())
            dist.recv(data.y, src=src_gpu)
            data.y = data.y.long()
            print(f"data.y.size: {data.y.size()}", flush=True)

            g_loc = torch.sparse_coo_tensor(g_loc_indices, g_loc_values, \
                                                size=torch.Size([g_loc_meta[1], g_loc_meta[2]]))
            edge_count = g_loc_meta[3].item()
            if args.dataset == "ogbn-papers100M":
                train_idx_len = torch.cuda.LongTensor(1).fill_(0)
                test_idx_len = torch.cuda.LongTensor(1).fill_(0)
                dist.recv(train_idx_len, src=src_gpu)
                dist.recv(test_idx_len, src=src_gpu)

                print(f"train_idx_len: {train_idx_len}", flush=True)
                train_idx = torch.cuda.LongTensor(train_idx_len.item()).fill_(0)
                test_idx = torch.cuda.LongTensor(test_idx_len.item()).fill_(0)
                dist.recv(train_idx, src=src_gpu)
                dist.recv(test_idx, src=src_gpu)
                print(f"recv train_idx: {train_idx}", flush=True)
                print(f"recv test_idx: {test_idx}", flush=True)
            g_loc = g_loc.cpu()
            ampbyp = split_am_partition(g_loc, partitions, rank, size, args.replication, args.normalize)
            vtx_count = g_loc.size(0)
            # g_loc = g_loc.cuda().coalesce().t_()
            # g_loc = g_loc.coalesce().t_().cuda()
            features_loc = inputs.to(device)
            print(f"g_loc.size: {g_loc.size()} g_loc._nnz: {g_loc._nnz()}", flush=True)
            print(f"features_loc.size: {features_loc.size()}", flush=True)

    else:
        features_partition, g_loc_partitions, ampbyp = one5d_partition(rank, size, inputs, adj_matrix, data, \
                                                                  inputs, num_classes, args.replication, \
                                                                  args.normalize, device, partitions)
        features_loc = features_partition[rank_c]
        g_loc = g_loc_partitions[rank_c]

    # one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, normalize, device)

    print("why segfault?", flush=True)
    features_loc = features_loc.to(device)
    print(f"g_loc: {g_loc}", flush=True)
    del g_loc
    g_loc = None

    # g_loc = g_loc.to(device)
    print("reached here", flush=True)
    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)

    print("create GCN model")
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
                      args.replication,
                      device,
                      row_groups=row_groups,
                      col_groups=col_groups)

    # communicate the indices before:
    row_procs = []
    for i in range(0, size, args.replication):
        row_procs.append(list(range(i, i + args.replication)))

    col_procs = []
    for i in range(args.replication):
        col_procs.append(list(range(i, size, args.replication)))

    counts_send = []
    row_indices_send = []

    print("pre send indices")
    counts_recv = [torch.cuda.LongTensor(1, 1, device=device).fill_(0) for i in range(size)]

    for i in range(len(ampbyp)):
        print(f"rank: {rank} i: {i}")

        unique_cols = ampbyp[i]._indices()[1].unique()
        for j in range(args.replication):
            row_indices_send.append(unique_cols)
            counts_send.append(torch.cuda.LongTensor([unique_cols.size()], device=device).resize_(1, 1))
    
    model.row_indices_send = row_indices_send

    print("all to all counts")  
    dist.all_to_all(counts_recv, counts_send)
 
    row_indices_recv = [torch.cuda.LongTensor(device=device).resize_(counts_recv[i].int().item(),).fill_(0) for i in range(len(counts_recv))]

    # row_data_recv = [torch.cuda.FloatTensor(device=self.device).resize_(counts_send[i].int().item(), inputs.size(1)).fill_(0) for i in range(len(counts_send))]
    # start = time.time()

    print("all to all indices")
    dist.all_to_all(row_indices_recv, row_indices_send)

    model.row_indices_recv = row_indices_recv
    print("all to all done")

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_per_proc = math.ceil(float(vtx_count) / (size / args.replication))

    rank_train_mask = []
    labels_rank = []

    if "ogb" not in args.dataset:
        train_nid = data.train_mask.nonzero().squeeze()
        if partitions:
            rank_train_mask = torch.split(data.train_mask, partitions, dim=0)[rank_c]
            labels_rank = torch.split(data.y, partitions, dim=0)[rank_c]
        else:
            print(data.train_mask.size())
            rank_train_mask = torch.split(data.train_mask, n_per_proc, dim=0)[rank_c]
            labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank_c]
    else:
        train_mask = torch.zeros(vtx_count).cuda()
        train_mask = train_mask.scatter_(0, train_idx, True)
        train_nid = train_idx
        if partitions:
            rank_train_mask = torch.split(train_mask, partitions, dim=0)[rank_c]
            labels_rank = torch.split(data.y, partitions, dim=0)[rank_c]
        else:
            rank_train_mask = torch.split(train_mask, n_per_proc, dim=0)[rank_c]
            labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank_c]
  


    # rank_train_mask = torch.split(data.train_mask, n_per_proc, dim=0)[rank_c]
    # rank_test_mask = torch.split(data.test_mask, n_per_proc, dim=0)[rank_c]
    # labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank_c]
    rank_train_nids = rank_train_mask.nonzero().squeeze()
    # rank_test_nids = rank_test_mask.nonzero().squeeze()

    # test_nid = data.test_mask.nonzero().squeeze()

    torch.manual_seed(0)
    total_start = time.time()

    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}", flush=True)
        if epoch == 1:
            total_start = time.time()
        model.train()

        # forward
        logits = model(g_loc, features_loc, ampbyp, epoch)
        loss = CAGF.cross_entropy(logits[rank_train_nids], labels_rank[rank_train_nids].long(), \
                                        train_nid.size(0), num_classes, partitioning, rank_c, col_groups[0], \
                                        size // args.replication)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        """
        # acc = evaluate(model, g_loc, features_loc, labels, val_nid, ampbyp, ampbyp_dgl, degrees, col_groups[0])
        # acc = evaluate(model, g_loc, features_loc, labels, val_nid, \
        acc = evaluate(model, g_loc, features_loc, labels_rank, rank_val_nids, \
                            val_mask.nonzero().squeeze().size(0), ampbyp, ampbyp_dgl, degrees, col_groups[0])
        print("Rank: {:05d} | Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(rank, epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000), flush=True)
        """
    dist.barrier()
    total_stop = time.time()
    print(f"total_time: {total_stop - total_start}")
    print(f"rank: {rank} timings: {model.timings}")
    """
    # print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    barrier_start = time.time()
    dist.barrier()
    model.timings["total"] += time.time() - total_start
    model.timings["barrier"] += time.time() - barrier_start
    print()
    # # acc = evaluate(model, g_loc, features_loc, labels, test_nid, ampbyp, ampbyp_dgl, degrees, col_groups[0])
    # acc = evaluate(model, g_loc, features_loc, labels, test_nid, \
    acc = evaluate(model, g_loc, features_loc, labels_rank, rank_test_nids, \
                        test_mask.nonzero().squeeze().size(0), ampbyp, ampbyp_dgl, degrees, col_groups[0])
    print("Test Accuracy {:.4f}".format(acc))

    print(flush=True)
    dist.barrier()
    print("Timings")
    model.timings["comp"] = model.timings["scomp"] + model.timings["dcomp"]
    model.timings["comm"] = model.timings["bcast"] + model.timings["reduce"] + model.timings["op"]
    # print("rank, total, scomp, dcomp, bcast, reduce, op, barrier")
    # print(f"{rank}, {model.timings['total']}, {model.timings['scomp']}, {model.timings['dcomp']}, {model.timings['bcast']}, {model.timings['reduce']}, {model.timings['op']}, {model.timings['barrier']}")
    print(f"rank: {rank} timings: {model.timings}")
    """
    print(f"rank: {rank} {logits}")


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
    parser.add_argument('--partitioning', default='ONE5D', type=str,
                            help='partitioning strategy to use')
    parser.add_argument('--replication', default=1, type=int,
                            help='partitioning strategy to use')
    parser.add_argument('--timers', action="store_true",
                            help='turn on timers')
    parser.add_argument('--partitions', default='', type=str,
                            help='text file for unequal partitions')
    args = parser.parse_args()
    print(args)

    main(args)
