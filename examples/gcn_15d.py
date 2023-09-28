import argparse
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
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
from cagnet.samplers import ladies_sampler, sage_sampler
from cagnet.samplers.utils import *
import cagnet.nn.functional as CAGF
import torch.nn.functional as F

from sparse_coo_tensor_cpp import sort_dst_proc_gpu
import quiver

import socket

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, aggr, rank, size, partitioning, replication, 
                                        device, group=None, row_groups=None, col_groups=None):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.aggr = aggr
        self.rank = rank
        self.size = size
        self.group = group
        self.row_groups = row_groups
        self.col_groups = col_groups
        self.device = device
        self.partitioning = partitioning
        self.replication = replication
        self.timings = dict()

        self.timings["total"] = []
        self.timings["sample"] = []
        self.timings["extract"] = []
        self.timings["extract-select"] = []
        self.timings["extract-inst"] = []
        self.timings["extract-coalesce"] = []
        self.timings["train"] = []
        self.timings["selectfeats"] = []
        self.timings["fwd"] = []
        self.timings["bwd"] = []
        self.timings["loss"] = []
        self.timings["fakemats"] = []

        self.timings["precomp"] = []
        self.timings["spmm"] = []
        self.timings["gemm_i"] = []
        self.timings["gemm_w"] = []
        self.timings["aggr"] = []

        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, self.partitioning, self.device))
        # hidden layers
        for i in range(n_layers - 2):
                self.layers.append(GCNConv(n_hidden, n_hidden, self.partitioning, self.device))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_classes, self.partitioning, self.device))
        # # self.layers.append(GCNConv(in_feats, n_classes, self.partitioning, self.device))

        # self.layers.append(SAGEConv(in_feats, n_hidden, root_weight=False, bias=False))
        # for _ in range(n_layers - 2):
        #     self.layers.append(SAGEConv(n_hidden, n_hidden, root_weight=False, bias=False))
        # self.layers.append(SAGEConv(n_hidden, n_classes, root_weight=False, bias=False))

    def forward(self, graphs, inputs, epoch):
        h = inputs
        for l, layer in enumerate(self.layers):
            # nnz_index = graphs[l]._values().nonzero().squeeze() # SAGEConv
            # edge_index = graphs[l]._indices()[:, nnz_index] # SAGEConv
            # edge_index = graphs[l] # SAGEConvfake
            # edge_index, _, size = graphs[l] # quiver
            # h = self.layers[l](h, edge_index) # SAGEConv
            h = layer(self, graphs[l], h, epoch) # GCNConv
            if l != len(self.layers) - 1:
                # h = CAGF.relu(h, self.partitioning)
                h = F.relu(h)

        # h = CAGF.log_softmax(self, h, self.partitioning, dim=1)
        h = F.log_softmax(h, dim=1)
        return h

    @torch.no_grad()
    def evaluate(self, graph, features, test_idx, labels):
        # subgraph_loader = NeighborSampler(graph, node_idx=None,
        #                                   sizes=[-1], batch_size=2048,
        #                                   shuffle=False, num_workers=6)

        # for i in range(self.n_layers):
        #     xs = []
        #     for batch_size, n_id, adj in subgraph_loader:
        #         edge_index, _, size = adj.to(self.device)
        #         x = features[n_id].to(self.device)
        #         # edge_index, _, size = adj
        #         # x = features[n_id]
        #         # x_target = x[:size[1]]
        #         # x = self.convs[i]((x, x_target), edge_index)
        #         x = self.layers[i](x, edge_index)
        #         if i != self.n_layers - 1:
        #             x = F.relu(x)
        #         # xs.append(x)
        #         xs.append(x[:batch_size])

        #     features = torch.cat(xs, dim=0)

        # return features

        subgraph_loader = NeighborSampler(graph, node_idx=None,
                                          sizes=[-1], batch_size=2048,
        # subgraph_loader = NeighborSampler(graph, node_idx=test_idx,
                                          # sizes=[-1, -1], batch_size=512,
                                          shuffle=False)
        non_eval_timings = copy.deepcopy(self.timings)
        for l, layer in enumerate(self.layers):
            hs = []
            for batch_size, n_id, adj in subgraph_loader:
                # edge_index, _, size = adj.to(self.device)
                edge_index, _, size = adj
                adj_batch = torch.sparse_coo_tensor(edge_index, 
                                                        torch.FloatTensor(edge_index.size(1)).fill_(1.0),
                                                        size)
                adj_batch = adj_batch.t().coalesce()
                h_batch = features[n_id]

                h = layer(self, adj_batch, h_batch, epoch=-1) # GCNConv
                if l != len(self.layers) - 1:
                    h = CAGF.relu(h, self.partitioning)
                hs.append(h) # GCNConv
                # hs.append(h[:batch_size]) # SAGEConv
            features = torch.cat(hs, dim=0)
        return features

        # print(f"test_idx: {test_idx}", flush=True)
        # non_eval_timings = copy.deepcopy(self.timings)
        # correct_count = 0
        # for batch_size, n_id, adjs in subgraph_loader:
        #     h = features[n_id]
        #     for l, (edge_index, _, size) in enumerate(adjs):
        #         # h = layer(self, adj_batch, h_batch, epoch=-1) # GCNConv
        #         h = self.layers[l](h, edge_index) # SAGEConv
        #         if l != len(self.layers) - 1:
        #             h = F.relu(h)
        #     h = F.log_softmax(h, dim=1)
        #     batch = n_id[:batch_size]
        #     print(f"batch: {batch}", flush=True)
        #     print(f"h[:batch_size]: {h[:batch_size]}", flush=True)
        #     print(f"h[:batch_size].argmax(dim=-1): {h[:batch_size].argmax(dim=-1)}", flush=True)
        #     preds = h[:batch_size].argmax(dim=-1) == labels[batch]
        #     correct_count += preds.sum()
        # return correct_count

        # """ SAGEConv inference """
        # subgraph_loader = NeighborSampler(graph, node_idx=None,
        #                                   sizes=[-1], batch_size=2048,
        #                                   shuffle=False)

        # # Compute representations of nodes layer by layer, using *all*
        # # available edges. This leads to faster computation in contrast to
        # # immediately computing the final representations of each batch.
        # total_edges = 0
        # for i in range(self.n_layers):
        #     xs = []
        #     for batch_size, n_id, adj in subgraph_loader:
        #         edge_index, _, size = adj.to(self.device)
        #         total_edges += edge_index.size(1)
        #         x = features[n_id].to(self.device)
        #         # x_target = x[:size[1]]
        #         # x = self.convs[i]((x, x_target), edge_index)
        #         x = self.layers[i](x, edge_index) # SAGEConv
        #         if i != self.n_layers - 1:
        #             x = F.relu(x)
        #         x = x[:batch_size]
        #         xs.append(x)
        #     features = torch.cat(xs, dim=0)

        # return features

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
    deg = torch.histc(adj_matrix[0].float(), bins=node_count)
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
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sparse.sum(mx, 1)
    r_inv = torch.float_power(rowsum, -1).flatten()
    # r_inv._values = r_inv._values()[torch.isinf(r_inv._values())] = 0.
    # r_mat_inv = torch.diag(r_inv._values())
    r_inv_values = torch.cuda.DoubleTensor(r_inv.size(0)).fill_(0)
    r_inv_values[r_inv._indices()[0,:]] = r_inv._values()
    # r_inv_values = r_inv._values()
    r_inv_values[torch.isinf(r_inv_values)] = 0
    r_mat_inv = torch.sparse_coo_tensor([np.arange(0, r_inv.size(0)).tolist(),
                                     np.arange(0, r_inv.size(0)).tolist()],
                                     r_inv_values,
                                     size=(r_inv.size(0), r_inv.size(0)))
    # mx = r_mat_inv.mm(mx.float())
    mx_indices, mx_values = torch_sparse.spspmm(r_mat_inv._indices(), r_mat_inv._values(), 
                                                    mx._indices(), mx._values(),
                                                    r_mat_inv.size(0), r_mat_inv.size(1), mx.size(1),
                                                    coalesced=True)
    mx = torch.sparse_coo_tensor(indices=mx_indices, values=mx_values.double(), size=(r_mat_inv.size(0), mx.size(1)))
    return mx

def one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, \
                            normalize):
    node_count = inputs.size(0)
    # n_per_proc = math.ceil(float(node_count) / size)
    # n_per_proc = math.ceil(float(node_count) / (size / replication))
    n_per_proc = node_count // (size // replication)

    am_partitions = None
    am_pbyp = None

    # inputs = inputs.to(torch.device("cpu"))
    # adj_matrix = adj_matrix.to(torch.device("cpu"))
    # torch.cuda.synchronize()

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)
        torch.cuda.synchronize()
        # proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        # am_pbyp, _ = split_coo(am_partitions[rank_c], node_count, n_per_proc, 0)
        # print(f"before", flush=True)
        # for i in range(len(am_pbyp)):
        #     if i == size // replication - 1:
        #         last_node_count = vtx_indices[i + 1] - vtx_indices[i]
        #         am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
        #                                                 size=(last_node_count, proc_node_count),
        #                                                 requires_grad=False)

        #         am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
        #                                         vtx_indices[rank_c], normalize)
        #     else:
        #         am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
        #                                                 size=(n_per_proc, proc_node_count),
        #                                                 requires_grad=False)

        #         am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
        #                                         vtx_indices[rank_c], normalize)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)).double(), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i], \
                                                    normalize)

        # input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)
        input_partitions = torch.split(inputs, inputs.size(0) // (size // replication), dim=0)
        if len(input_partitions) > (size // replication):
            input_partitions_fused = [None] * (size // replication)
            input_partitions_fused[:-1] = input_partitions[:-2]
            input_partitions_fused[-1] = torch.cat(input_partitions[-2:], dim=0)
            input_partitions = input_partitions_fused

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_partitions, input_partitions

def one5d_partition_mb(rank, size, batches, replication, mb_count):
    rank_c = rank // replication
    # batch_partitions = torch.split(batches, math.ceil(float(mb_count / (size / replication))), dim=0)
    batch_partitions = torch.split(batches, int(mb_count // (size / replication)), dim=0)
    return batch_partitions[rank_c]

def main(args, batches=None):
    # load and preprocess dataset
    # Initialize distributed environment with SLURM
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"
    
    print(f"device_count: {torch.cuda.device_count()}")
    print(f"hostname: {socket.gethostname()}", flush=True)
    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend)
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}", flush=True)
    torch.cuda.set_device(rank % args.gpu)

    device = torch.device(f'cuda:{rank % args.gpu}')

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_inner_timer = torch.cuda.Event(enable_timing=True)
    stop_inner_timer = torch.cuda.Event(enable_timing=True)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)

    if args.dataset == "cora" or args.dataset == "reddit":
        if args.dataset == "cora":
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset == "reddit":
            dataset = Reddit(path)

        data = dataset[0]
        data = data.to(device)
        # data.x.requires_grad = True
        inputs = data.x.to(device)
        # inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        adj_matrix = edge_index
    elif args.dataset == "Amazon":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("/global/homes/a/alokt/data/Amazon/processed/data.pt")
        print(f"Done loading coo", flush=True)
        # n = 14249639
        n = 9430088
        num_features = 128
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        # data.train_mask = torch.ones(n).long()
        data.train_mask = torch.zeros(n).long()
        data.train_mask[:args.batch_size * 1024] = 1
        data.test_mask = torch.zeros(n).long()
        data.test_mask[args.batch_size * args.n_bulkmb:] = 1
        adj_matrix = edge_index.t_()
        data = data.to(device)
        # inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein":
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../../data/protein/processed/protein.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542
        num_features = 128
        num_classes = 256
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        # data.train_mask = torch.ones(n).long()
        data.train_mask = torch.zeros(n).long()
        data.train_mask[:args.batch_size * 2048] = 1
        data.test_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        # inputs.requires_grad = True
        data.y = data.y.to(device)
    elif args.dataset == "Protein_sg2":
        n = 17491085
        num_features = 128
        num_classes = 256
        if rank % args.gpu == 0:
            print(f"Loading coo...", flush=True)
            edge_index = torch.load("../../data/protein_sg2/processed/protein_sg2.pt")
            print(f"Done loading coo", flush=True)
            inputs = torch.rand(n, num_features)
            adj_matrix = edge_index.t_()
            # inputs.requires_grad = True
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        data.test_mask = torch.ones(n).long()
        data = data.to(device)
        data.y = data.y.to(device)
    elif args.dataset.startswith("ogb"):
        import ogb
        data = Data()
        from ogb.nodeproppred import PygNodePropPredDataset # only import if necessary, takes a long time
        if ("papers100M" in args.dataset and rank % args.gpu == 0) or "products" in args.dataset:
            dataset = PygNodePropPredDataset(name=args.dataset, root="/global/u1/a/alokt/data")
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

    # print(f"adj_matrix.size(): {adj_matrix.size()}")
    # print(f"adj_matrix: {adj_matrix}")

    # adj_matrix, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))
    # if args.sample_method == "ladies":
    #     print(f"before adj_matrix.size: {adj_matrix.size()}", flush=True)
    #     adj_matrix, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))
    #     print(f"after adj_matrix.size: {adj_matrix.size()}", flush=True)
    if args.dataset != "ogbn-papers100M" and args.dataset != "Protein_sg2":
        edge_count = adj_matrix.size(1)

    partitioning = Partitioning.ONE5D

    row_groups, col_groups = get_proc_groups(rank, size, args.replication)

    proc_row = size // args.replication
    rank_row = rank // args.replication
    rank_col = rank % args.replication
    if rank_row >= (size // args.replication):
        return

    print("start partitioning", flush=True)

    torch.cuda.synchronize()
    if args.dataset == "ogbn-papers100M" or "Protein" in args.dataset:
        if rank % args.gpu == 0:
            inputs = inputs.to(torch.device("cpu"))
            adj_matrix = adj_matrix.to(torch.device("cpu"))
            # send g_loc.nnz, g_loc, train_idx.len, and train_idx
            features_loc, g_loc, am_partitions, input_partitions = \
                    one5d_partition(rank, size, inputs, adj_matrix, data, 
                                        inputs, num_classes, args.replication, \
                                        args.normalize)

            for i in range(1, args.gpu):
                print(f"iteration i: {i}", flush=True)
                rank_send_row = (rank + i) // args.replication
                dst_gpu = rank + i
                g_send = am_partitions[rank_send_row]
                print("coalescing", flush=True)
                g_send = g_send.coalesce().t_()
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
                    print(f"train_idx_len: {train_idx_len} train_idx: {train_idx}", flush=True)
                    print(f"test_idx_len: {test_idx_len} test_idx: {test_idx}", flush=True)
                    dist.send(train_idx, dst=dst_gpu)
                    dist.send(test_idx, dst=dst_gpu)

            # features_loc, g_loc, _, _ = one5d_partition(rank, size, inputs, adj_matrix, data, inputs, num_classes, \
            #                                 args.replication, args.normalize)
            print("coalescing", flush=True)
            g_loc = g_loc.coalesce().t_()
            print("normalizing", flush=True)
            g_loc = g_loc.to(device)
            features_loc = features_loc.to(device)
            g_loc = g_loc.double()
            edge_count = adj_matrix.size(1)

            # del adj_matrix # Comment when testing
            # del inputs
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
            if rank_row < proc_row - 1:
                inputs = torch.cuda.FloatTensor(g_loc_meta[1].item(), num_features)
            else:
                rows = g_loc_meta[1].item() + (g_loc_meta[2].item() - proc_row * g_loc_meta[1].item())
                print(f"g_loc_meta: {g_loc_meta} rows: {rows}", flush=True)
                inputs = torch.cuda.FloatTensor(rows, num_features)
            dist.recv(inputs, src=src_gpu)

            # data = Data()
            data.y = torch.cuda.FloatTensor(g_loc_meta[2].item())
            dist.recv(data.y, src=src_gpu)
            data.y = data.y.long()

            g_loc = torch.sparse_coo_tensor(g_loc_indices, g_loc_values, \
                                                size=torch.Size([g_loc_meta[1], g_loc_meta[2]]))
            edge_count = g_loc_meta[3].item()

            if args.dataset == "ogbn-papers100M":
                train_idx_len = torch.cuda.LongTensor(1).fill_(0)
                test_idx_len = torch.cuda.LongTensor(1).fill_(0)
                dist.recv(train_idx_len, src=src_gpu)
                dist.recv(test_idx_len, src=src_gpu)

                train_idx = torch.cuda.LongTensor(train_idx_len.item()).fill_(0)
                test_idx = torch.cuda.LongTensor(test_idx_len.item()).fill_(0)
                dist.recv(train_idx, src=src_gpu)
                dist.recv(test_idx, src=src_gpu)
                print(f"recv train_idx: {train_idx}", flush=True)
                print(f"recv test_idx: {test_idx}", flush=True)
            torch.cuda.synchronize()
            features_loc = inputs.to(device)
    else:
        inputs = inputs.to(torch.device("cpu"))
        adj_matrix = adj_matrix.to(torch.device("cpu"))
        features_loc, g_loc, _, _ = one5d_partition(rank, size, inputs, adj_matrix, data, inputs, num_classes, \
                                                    args.replication, args.normalize)
        print("coalescing", flush=True)
        g_loc = g_loc.coalesce().t_()
        print("normalizing", flush=True)
        g_loc = g_loc.to(device)
        g_loc = g_loc.double()
        # features_loc = inputs
        features_loc = features_loc.to(device)
        # del inputs

    print("end partitioning", flush=True)
    print(f"g_loc.nnz: {g_loc._nnz()}", flush=True)
    print(f"features_loc.size: {features_loc.size()}", flush=True)
    dist.barrier()

    if rank == 0:
        adj_matrix = adj_matrix.cpu()
        inputs = inputs.cpu()

    # g_loc_indices, _ = add_remaining_self_loops(g_loc._indices(), num_nodes=g_loc.size(0))
    # g_loc_values = torch.cuda.DoubleTensor(g_loc_indices.size(1)).fill_(1)
    # g_loc = torch.sparse_coo_tensor(g_loc_indices, g_loc_values, g_loc.size())
    # g_loc = row_normalize(g_loc)
    print("done normalizing", flush=True)
    torch.set_printoptions(precision=10)

    # n_per_proc = math.ceil(float(g_loc.size(0)) / (size / args.replication))
    n_per_proc = g_loc.size(0) // (size // args.replication)

    if not args.dataset.startswith("ogb"):
        train_nid = data.train_mask.nonzero().squeeze()
        test_nid = data.test_mask.nonzero().squeeze()
        print(f"train_nid.size: {train_nid.size()}")
        print(f"test_nid.size: {test_nid.size()}")
    else:
        train_nid = train_idx
        test_nid = test_idx
        print(f"train_nid.size: {train_nid.size()}")
        print(f"test_nid.size: {test_nid.size()}")
    
    # # # adj_matrix = None # Uncomment bottom for testing
    # adj_matrix = adj_matrix.cuda()
    # adj_matrix = torch.sparse_coo_tensor(adj_matrix, 
    #                                     torch.cuda.FloatTensor(adj_matrix.size(1)).fill_(1.0), 
    #                                     size=(node_count, node_count))
    # adj_matrix = adj_matrix.coalesce().cuda().double()

    # return frontiers, adj_matrices, adj_matrix
    # return frontiers, adj_matrices, adj_matrix, col_groups

    # create GCN model
    torch.manual_seed(0)
    model = GCN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.n_layers,
                      args.aggr,
                      rank,
                      size,
                      Partitioning.NONE,
                      args.replication,
                      device,
                      row_groups=row_groups,
                      col_groups=col_groups)
    model = model.to(device)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dur = []
    # torch.manual_seed(0)
    # tally = torch.cuda.IntTensor(size).fill_(0)
    tally = torch.cuda.IntTensor(proc_row).fill_(0)
    row_n_bulkmb = int(args.n_bulkmb / (size / args.replication))
    if rank == size - 1:
        row_n_bulkmb = args.n_bulkmb - row_n_bulkmb * (proc_row - 1)

    rank_n_bulkmb = int(row_n_bulkmb / args.replication)
    if rank == size - 1:
        rank_n_bulkmb = row_n_bulkmb - rank_n_bulkmb * (args.replication - 1)

    print(f"rank_n_bulkmb: {rank_n_bulkmb}")
    # g_loc = g_loc.cpu()
    g_loc = g_loc.to_sparse_csr()
    g_loc = g_loc.to(device)

    csr_topo = quiver.CSRTopo(adj_matrix)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [args.samp_num] * args.n_layers, 0, 
                                                    mode='UVA')
    # train_loader = torch.utils.data.DataLoader(train_nid, batch_size=args.batch_size, 
    #                                                 shuffle=True, drop_last=True)
    for epoch in range(args.n_epochs):
        # train_loader_iter = iter(train_loader)
        print(f"Epoch: {epoch}", flush=True)
        if epoch >= 1:
            epoch_start = time.time()
        model.train()

        feat_dims = []
        adj0_dims = []
        adj1_dims = []
        adj0_nnzs = []
        adj1_nnzs = []
        adj_nnzs = []

        # print("Constructing batches", flush=True)
        torch.cuda.nvtx.range_push("nvtx-construct-batches")
        batch_count = -(train_nid.size(0) // -args.batch_size) # ceil(train_nid.size(0) / batch_size)
        # if batches is None:
        torch.manual_seed(epoch)
        vertex_perm = torch.randperm(train_nid.size(0))
        batches_all = train_nid[vertex_perm]
        torch.cuda.nvtx.range_pop() # construct-batches

        torch.cuda.nvtx.range_push("nvtx-extract-batches")
        last_batch = False
        for b in range(0, batch_count, args.n_bulkmb):
            if b + args.n_bulkmb > batch_count:
                break
                last_batch = True
                tmp_bulkmb = args.n_bulkmb
                tmp_batch_size = args.batch_size
                args.n_bulkmb = 1
                args.batch_size = batch_count - b
            batches = batches_all[b:(b + args.n_bulkmb * args.batch_size)].view(args.n_bulkmb, args.batch_size)
            batches_loc = one5d_partition_mb(rank, size, batches, args.replication, args.n_bulkmb)

            batches_indices_rows = torch.arange(batches_loc.size(0)).to(device)
            batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
            batches_indices_cols = batches_loc.view(-1)
            batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
        if epoch >= 1:
            epoch_start = time.time()
        torch.cuda.nvtx.range_pop() # extract-batches

        # torch.cuda.nvtx.range_push("nvtx-construct-batches")
        # batch_count = -(train_nid.size(0) // -args.batch_size) # ceil(train_nid.size(0) / batch_size)
        # # print(f"batch_count: {batch_count}", flush=True)
        # if batches is None:
        #     torch.manual_seed(epoch)
        #     vertex_perm = torch.randperm(train_nid.size(0))
        #     batches_all = train_nid[vertex_perm]
        # torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-bulk")
        last_batch = False
        for b in range(0, batch_count, args.n_bulkmb):
            if b + args.n_bulkmb > batch_count:
                break
                last_batch = True
                tmp_bulkmb = args.n_bulkmb
                tmp_batch_size = args.batch_size
                args.n_bulkmb = 1
                args.batch_size = batch_count - b
            torch.cuda.nvtx.range_push("nvtx-partition-batches")
            batches = batches_all[b:(b + args.n_bulkmb * args.batch_size)].view(args.n_bulkmb, args.batch_size)
            batches_loc = one5d_partition_mb(rank, size, batches, args.replication, args.n_bulkmb)

            batches_indices_rows = torch.arange(batches_loc.size(0)).to(device)
            batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
            batches_indices_cols = batches_loc.view(-1)
            batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
            batches_values = torch.cuda.DoubleTensor(batches_loc.size(1) * batches_loc.size(0)).fill_(1.0)
            batches_loc = torch.sparse_coo_tensor(batches_indices, batches_values, (batches_loc.size(0), g_loc.size(1)))
            torch.cuda.nvtx.range_pop() # partition-batches
            # g_loc = torch.pow(g_loc, 2)

            node_count = g_loc.size(1)
            # adj_matrix = adj_matrix.cuda()
            if args.n_darts == -1:
                avg_degree = int(edge_count / node_count)
                if args.sample_method == "ladies":
                    args.n_darts = avg_degree * args.batch_size * (args.replication ** 2)
                elif args.sample_method == "sage":
                    args.n_darts = avg_degree * 2

            # for sa-spgemm
            # nnz_row_masks = torch.cuda.BoolTensor((size // args.replication) * g_loc._indices().size(1)) 
            nnz_row_masks = torch.cuda.BoolTensor((size // args.replication) * g_loc._nnz()) 
            nnz_row_masks.fill_(0)
            
            nnz_recv_upperbound = edge_count // (size // args.replication)

            if epoch >= 1:
                start_time(start_timer)
            # print("Sampling", flush=True)
            torch.cuda.nvtx.range_push("nvtx-sampling")
            if args.sample_method == "ladies":
                current_frontier, next_frontier, adj_matrices_bulk = \
                                                ladies_sampler(g_loc, batches_loc, args.batch_size, \
                                                                            args.samp_num, args.n_bulkmb, \
                                                                            args.n_layers, args.n_darts, \
                                                                            args.semibulk, args.replication, \
                                                                            nnz_row_masks, rank, size, 
                                                                            row_groups, col_groups, args.timing)
            elif args.sample_method == "sage":
                # nnz_row_masks.fill_(False)
                frontiers_bulk, adj_matrices_bulk = sage_sampler(g_loc, batches_loc, args.batch_size, \
                                                                        args.samp_num, args.n_bulkmb, \
                                                                        args.n_layers, args.n_darts, \
                                                                        args.replication, nnz_row_masks, 
                                                                        rank, size, row_groups, 
                                                                        col_groups, args.timing, args.baseline)
                
            if epoch >= 1:
                model.timings["sample"].append(stop_time(start_timer, stop_timer))
            torch.cuda.nvtx.range_pop() # nvtx-sampling

            if epoch >= 1:
                start_time(start_timer)
            # g_loc = g_loc.coalesce()

            # print("Training", flush=True)
            torch.cuda.nvtx.range_push("nvtx-training")

            for i in range(rank_n_bulkmb):
                # forward
                # batch_vtxs = frontiers[0][i].view(-1)
                torch.cuda.nvtx.range_push("nvtx-extracting")

                batch_select_size = args.batch_size
                batch_select_min = i * args.batch_size 
                batch_select_max = (i + 1) * args.batch_size

                rank_n_bulkmb_row = rank_n_bulkmb // args.replication
                if rank_col == args.replication - 1:
                    rank_n_bulkmb_row = rank_n_bulkmb - rank_n_bulkmb_row * (args.replication - 1)

                batch_select_min += batch_select_size * rank_col * rank_n_bulkmb_row
                batch_select_max += batch_select_size * rank_col * rank_n_bulkmb_row
                batch_vtxs = frontiers_bulk[0][batch_select_min:batch_select_max,:].view(-1)

                # src_vtxs = frontiers[-1][i].view(-1)
                src_select_size = args.batch_size * (args.samp_num ** (args.n_layers - 1))
                src_select_min = i * args.batch_size * (args.samp_num ** (args.n_layers - 1))
                src_select_max = (i + 1) * args.batch_size  * (args.samp_num ** (args.n_layers - 1))
                
                src_select_min += src_select_size * rank_col * rank_n_bulkmb_row
                src_select_max += src_select_size * rank_col * rank_n_bulkmb_row
                src_vtxs = frontiers_bulk[-1][src_select_min:src_select_max,:].view(-1)
                # print(f"src_vtxs: {src_vtxs} src_vtxs.size: {src_vtxs.size()} zero_count: {(src_vtxs == 0).sum()}", flush=True)

                adjs = [None] * args.n_layers
                for l in range(args.n_layers):
                    adj_select_size = args.batch_size * (args.samp_num ** l)
                    adj_row_select_min = i * args.batch_size * (args.samp_num ** l)
                    adj_row_select_max = (i + 1) * args.batch_size  * (args.samp_num ** l)
                    adj_row_select_min += adj_select_size * rank_col * rank_n_bulkmb_row
                    adj_row_select_max += adj_select_size * rank_col * rank_n_bulkmb_row
                    adj_row_select_max = min(adj_row_select_max, adj_matrices_bulk[l].size(0))

                    if epoch >= 1:
                        start_time(start_inner_timer)

                    adj_nnz_start = adj_matrices_bulk[l].crow_indices()[adj_row_select_min]
                    adj_nnz_stop = adj_matrices_bulk[l].crow_indices()[adj_row_select_max]

                    adj_sample_crows = adj_matrices_bulk[l].crow_indices()[adj_row_select_min:(adj_row_select_max+1)].clone()
                    adj_sample_cols = adj_matrices_bulk[l].col_indices()[adj_nnz_start:adj_nnz_stop]
                    crows_start = adj_sample_crows[0].item()
                    adj_sample_crows -= crows_start

                    adj_sample_values = adj_matrices_bulk[l].values()[adj_nnz_start:adj_nnz_stop].float()

                    if epoch >= 1:
                        model.timings["extract-select"].append(stop_time(start_inner_timer, stop_inner_timer))

                    if epoch >= 1:
                        start_time(start_inner_timer)
                    if args.sample_method == "ladies":
                        adj_matrix_sample = torch.sparse_coo_tensor(adj_matrix_sample_indices, \
                                                        adj_matrix_sample_values, \
                                                        (args.batch_size, args.samp_num + args.batch_size))
                    else:
                        adj_sample_row_lens = adj_sample_crows[1:] - adj_sample_crows[:-1]
                        row_count = adj_row_select_max - adj_row_select_min
                        adj_sample_rows = torch.repeat_interleave(
                                            torch.arange(0, row_count, 
                                                            device=adj_sample_row_lens.device), 
                                            adj_sample_row_lens)
                        adj_sample_indices = torch.stack((adj_sample_rows, adj_sample_cols))
                        adj_matrix_sample = torch.sparse_coo_tensor(adj_sample_indices, \
                                                        adj_sample_values, \
                                                        (args.batch_size * (args.samp_num ** l), \
                                                                args.batch_size * (args.samp_num ** (l + 1))))
                        adj_nnzs.append(adj_sample_values.size(0))
                        # print(f"i: {i} sample.nnz: {adj_matrix_sample._nnz()} sample.size: {adj_matrix_sample.size()} emptyrows: {(adj_sample_row_lens == 0).sum()}")
                    if epoch >= 1:
                        model.timings["extract-inst"].append(stop_time(start_inner_timer, stop_inner_timer))

                    if epoch >= 1:
                        start_time(start_inner_timer)
                    # adjs[args.n_layers - l - 1] = adj_matrix_sample.to_sparse_coo()
                    adjs[args.n_layers - l - 1] = adj_matrix_sample.coalesce()
                    if epoch >= 1:
                        model.timings["extract-coalesce"].append(stop_time(start_inner_timer, stop_inner_timer))

                torch.cuda.nvtx.range_pop() # nvtx-extracting
                if epoch >= 1:
                    start_time(start_inner_timer)
                torch.cuda.nvtx.range_push("nvtx-selectfeats")
                if size > 1:
                    src_vtxs_sort = torch.cuda.LongTensor(src_vtxs.size(0))
                    og_idxs = torch.cuda.LongTensor(src_vtxs.size(0))
                    tally.fill_(0)

                    # node_per_row = int(math.ceil(node_count / (size / args.replication)))
                    # node_per_proc = int(math.ceil(node_count / size))
                    node_per_row = node_count // (size // args.replication)
                    node_per_proc = node_count // size
                    # sort_dst_proc_gpu(src_vtxs, src_vtxs_sort, og_idxs, tally, node_per_proc)
                    sort_dst_proc_gpu(src_vtxs, src_vtxs_sort, og_idxs, tally, 
                                        node_per_row, proc_row)
                    src_vtx_per_proc = src_vtxs_sort.split(tally.tolist())
        
                    output_tally = []
                    for j in range(proc_row):
                        output_tally.append(torch.cuda.IntTensor(1))
                    input_tally = list(torch.split(tally, 1))

                    dist.all_to_all(output_tally, input_tally, col_groups[rank_col])

                    output_src_vtxs = torch.cuda.LongTensor(sum(output_tally).item())
                    dist.all_to_all_single(output_src_vtxs, src_vtxs_sort, 
                                                output_tally, input_tally, 
                                                col_groups[rank_col])
                    output_src_vtxs -= node_per_row * rank_row
                    # output_src_vtxs -= node_per_proc * rank

                    input_features = features_loc[output_src_vtxs]
                    output_features = torch.cuda.FloatTensor(src_vtxs.size(0), features_loc.size(1))
                    dist.all_to_all_single(output_features, input_features, 
                                                input_tally, output_tally, 
                                                col_groups[rank_col])

                    features_batch = torch.cuda.FloatTensor(output_features.size())
                    features_batch[og_idxs] = output_features
                else:
                    features_batch = features_loc[src_vtxs]

                torch.cuda.nvtx.range_pop() # nvtx-selectfeats
                if epoch >= 1:
                    model.timings["selectfeats"].append(stop_time(start_inner_timer, stop_inner_timer))

                # if epoch >= 1:
                #     start_time(start_inner_timer)
                # torch.cuda.nvtx.range_push("nvtx-fakemats")
                # features_batch = torch.cuda.FloatTensor(69740, 128) # SAGEConvfake
                # adjs0_rows = torch.randint(0, 9862, (91518,)).cuda().long() # SAGEConvfake
                # adjs0_cols = torch.randint(0, 69740, (91518,)).cuda().long() # SAGEConvfake
                # adjs[0] = torch.stack((adjs0_rows, adjs0_cols)) # SAGEConvfake
                # adjs1_rows = torch.randint(0, 1024, (9065,)).cuda().long() # SAGEConvfake
                # adjs1_cols = torch.randint(0, 9862, (9065,)).cuda().long() # SAGEConvfake
                # adjs[1] = torch.stack((adjs1_rows, adjs1_cols)) # SAGEConvfake

                # features_batch = torch.cuda.FloatTensor(69740, 128) # GCNConvfake
                # adjs0_rows = torch.randint(0, 9862, (91518,)).cuda().long() # GCNConvfake
                # adjs0_cols = torch.randint(0, 69740, (91518,)).cuda().long() # GCNConvfake
                # adjs0_indices = torch.stack((adjs0_rows, adjs0_cols)) # GCNConvfake
                # adjs0_vals = torch.cuda.FloatTensor(91518).fill_(1.0) # GCNConvfake
                # adjs[0] = torch.sparse_coo_tensor(adjs0_indices, adjs0_vals, torch.Size([9862, 69740])) # GCNConvfake
                # adjs[0] = adjs[0].coalesce()
                # adjs1_rows = torch.randint(0, 1024, (9065,)).cuda().long() # GCNConvfake
                # adjs1_cols = torch.randint(0, 9862, (9065,)).cuda().long() # GCNConvfake
                # adjs1_indices = torch.stack((adjs1_rows, adjs1_cols)) # GCNConvfake
                # adjs1_vals = torch.cuda.FloatTensor(9065).fill_(1.0) # GCNConvfake
                # adjs[1] = torch.sparse_coo_tensor(adjs1_indices, adjs1_vals, torch.Size([1024, 9862])) # GCNConvfake
                # adjs[1] = adjs[1].coalesce()
                # if epoch >= 1:
                #     model.timings["fakemats"].append(stop_time(start_inner_timer, stop_inner_timer))
                # torch.cuda.nvtx.range_pop() # nvtx-fakemats

                if epoch >= 1:
                    start_time(start_inner_timer)
                torch.cuda.nvtx.range_push("nvtx-fwd")
                # feat_dims.append(features_batch.size(0))
                # adj0_dims.append(adjs[0].size(1))
                # adj1_dims.append(adjs[1].size(1))
                # adj0_nnzs.append(adjs[0]._nnz())
                # adj1_nnzs.append(adjs[1]._nnz())

                # seeds = next(train_loader_iter)
                # n_id, batch_size, adjs = quiver_sampler.sample(batch_vtxs)
                # adjs = [adj.to(rank) for adj in adjs]
                # features_batch = features_loc[n_id]
                optimizer.zero_grad()
                logits = model(adjs, features_batch, epoch)
                torch.cuda.nvtx.range_pop() # nvtx-fwd
                if epoch >= 1:
                    model.timings["fwd"].append(stop_time(start_inner_timer, stop_inner_timer))

                if epoch >= 1:
                    start_time(start_inner_timer)
                torch.cuda.nvtx.range_push("nvtx-loss")
                print(f"logits: {logits[:args.batch_size]}", flush=True)
                loss = F.nll_loss(logits, data.y[batch_vtxs].long()) # GCNConv
                # loss = F.nll_loss(logits[:args.batch_size], data.y[batch_vtxs].long()) # SAGEConv
                # loss = F.nll_loss(logits[:args.batch_size], data.y[seeds].long()) # quiver

                torch.cuda.nvtx.range_pop() # nvtx-loss
                if epoch >= 1:
                    model.timings["loss"].append(stop_time(start_inner_timer, stop_inner_timer))
                
                if epoch >= 1:
                    start_time(start_inner_timer)
                torch.cuda.nvtx.range_push("nvtx-bwd")
                loss.backward()
                torch.cuda.nvtx.range_pop() # nvtx-bwd
                if epoch >= 1:
                    model.timings["bwd"].append(stop_time(start_inner_timer, stop_inner_timer))

                for W in model.parameters():
                    dist.all_reduce(W.grad)
                torch.cuda.nvtx.range_push("nvtx-optstep")
                optimizer.step()
                torch.cuda.nvtx.range_pop() # nvtx-optstep
            if epoch >= 1:
                model.timings["train"].append(stop_time(start_timer, stop_timer))

            if last_batch:
                # break
                last_batch = False
                args.n_bulkmb = tmp_bulkmb 
                args.batch_size = tmp_batch_size
            torch.cuda.nvtx.range_pop() # nvtx-training
        torch.cuda.nvtx.range_pop() # nvtx-bulk

        dist.barrier()
        if epoch >= 1:
            dur.append(time.time() - epoch_start)
        if rank == 0 and epoch >= 1 and (epoch % 5 == 0 or epoch == args.n_epochs - 1):
            print("Evaluating", flush=True)
            # acc = model.evaluate(g_loc, features_loc, labels_rank, rank_val_nids, \
            #                     data.val_mask.nonzero().squeeze().size(0), col_groups[0])
            acc1 = 0.0
            acc3 = 0.0
            if args.timing:
                sample_dur = [x / 1000 for x in model.timings["sample"]]
                train_dur = [x / 1000 for x in model.timings["train"]]
                fwd_dur = [x / 1000 for x in model.timings["fwd"]]
                bwd_dur = [x / 1000 for x in model.timings["bwd"]]
                selectfeats_dur = [x / 1000 for x in model.timings["selectfeats"]]
                precomp_dur = [x / 1000 for x in model.timings["precomp"]]
                spmm_dur = [x / 1000 for x in model.timings["spmm"]]
                gemmi_dur = [x / 1000 for x in model.timings["gemm_i"]]
                gemmw_dur = [x / 1000 for x in model.timings["gemm_w"]]
                aggr_dur = [x / 1000 for x in model.timings["aggr"]]
                fakemats_dur = [x / 1000 for x in model.timings["fakemats"]]
                # sf_precomp_dur = [x / 1000 for x in model.timings["selectfeats-precomp"]]
                # sf_sortdst_dur = [x / 1000 for x in model.timings["selectfeats-sortdst"]]
                # sf_tallysplit_dur = [x / 1000 for x in model.timings["selectfeats-tallysplit"]]
                # sf_tallya2a_dur = [x / 1000 for x in model.timings["selectfeats-tallya2a"]]
                # sf_vtxsa2a_dur = [x / 1000 for x in model.timings["selectfeats-vtxsa2a"]]
                # sf_featsa2a_dur = [x / 1000 for x in model.timings["selectfeats-featsa2a"]]
                
                # extract_dur = [x / 1000 for x in model.timings["extract"]]
                extract_select_dur = [x / 1000 for x in model.timings["extract-select"]]
                extract_inst_dur = [x / 1000 for x in model.timings["extract-inst"]]
                extract_coalesce_dur = [x / 1000 for x in model.timings["extract-coalesce"]]

                print(f"sample: {np.sum(sample_dur)} extract-select: {np.sum(extract_select_dur)} extract-inst: {np.sum(extract_inst_dur)} extract-coalesce: {np.sum(extract_coalesce_dur)} train: {np.sum(train_dur)}", flush=True)
                print(f"feats: {np.sum(selectfeats_dur)} fwd: {np.sum(fwd_dur)} bwd: {np.sum(bwd_dur)}")
                print(f"precomp: {np.sum(precomp_dur)} spmm: {np.sum(spmm_dur)} gemmi: {np.sum(gemmi_dur)} gemmw: {np.sum(gemmw_dur)} aggr: {np.sum(aggr_dur)} fakemats: {np.sum(fakemats_dur)}")
                # print(f"feat_dim.median: {np.median(feat_dims)}")
                # print(f"adj0_dim.median: {np.median(adj0_dims)}")
                # print(f"adj1_dim.median: {np.median(adj1_dims)}")
                # print(f"adj0_nnz.median: {np.median(adj0_nnzs)}")
                # print(f"adj1_nnz.median: {np.median(adj1_nnzs)}")
                # print(f"adj_nnzs.median: {np.median(adj_nnzs)}")
                print(f"total: {np.sum(dur)}", flush=True)
            if args.dataset != "Amazon" and ("Protein" not in args.dataset):
                model = model.cpu()
                train_nid = train_nid.cpu()
                test_nid = test_nid.cpu()
                out = model.evaluate(adj_matrix, inputs, test_nid, data.y)
                # correct_count = model.evaluate(adj_matrix, inputs, test_nid, data.y.cpu())
                res = out.argmax(dim=-1) == data.y.cpu()
                # acc1 = int(res[train_nid].sum()) / train_nid.size(0)
                acc3 = int(res[test_nid].sum()) / test_nid.size(0)
                # acc3 = correct_count / test_nid.size(0)
                train_nid = train_nid.to(device)
                test_nid = test_nid.to(device)
                model = model.to(device)
                print("Rank: {:05d} | Epoch: {:05d} | Time(s): {:.4f} | Loss: {:.4f} | Accuracy: {:.4f}".format(rank, epoch, np.sum(dur), loss.item(), acc3), flush=True)
            else:
                print("Rank: {:05d} | Epoch: {:05d} | Time(s): {:.4f} | Accuracy: {:.4f}".format(rank, epoch, np.sum(dur), acc3), flush=True)

        dist.barrier() 
                
    total_stop = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--sample-method", type=str, default="ladies",
                        help="sampling algorithm for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=4,
                        help="gpus per node")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="number of vertices in minibatch")
    parser.add_argument("--samp-num", type=int, default=64,
                        help="number of vertices per layer of layer-wise minibatch")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggr", type=str, default="mean",
                        help="Aggregator type: mean/sum")
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
    parser.add_argument('--n-bulkmb', default=1, type=int,
                            help='number of minibatches to sample in bulk')
    parser.add_argument('--bulk-batch-fetch', default=1, type=int,
                            help='number of minibatches to fetch features for in bulk')
    parser.add_argument('--n-darts', default=-1, type=int,
                            help='number of darts to throw per minibatch in LADIES sampling')
    parser.add_argument('--semibulk', default=128, type=int,
                            help='number of batches to column extract from in bulk')
    parser.add_argument('--timing', action="store_true",
                            help='whether to turn on timers')
    parser.add_argument('--baseline', action="store_true",
                            help='whether to avoid col selection for baseline comparison')
    args = parser.parse_args()
    print(args)

    main(args)
