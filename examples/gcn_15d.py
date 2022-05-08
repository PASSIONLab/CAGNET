import argparse
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
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
from cagnet.samplers import ladies_sampler
import cagnet.nn.functional as CAGF

import socket

class GCN(nn.Module):
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, rank, size, partitioning, replication, device,
                    group=None, row_groups=None, col_groups=None):
    super(GCN, self).__init__()
    self.layers = nn.ModuleList()
    self.rank = rank
    self.size = size
    self.group = group
    self.row_groups = row_groups
    self.col_groups = col_groups
    self.device = device
    self.partitioning = partitioning
    self.replication = replication
    self.timings = dict()

    # input layer
    self.layers.append(GCNConv(in_feats, n_hidden, self.partitioning, self.device))
    # hidden layers
    for i in range(n_layers - 1):
        self.layers.append(GCNConv(n_hidden, n_hidden, self.partitioning, self.device))
    # output layer
    self.layers.append(GCNConv(n_hidden, n_classes, self.partitioning, self.device))

  def forward(self, graph, inputs, ampbyp):
    h = inputs
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
    r_inv_values = r_inv._values()
    r_inv_values[torch.isinf(r_inv_values)] = 0
    r_mat_inv = torch.sparse_coo_tensor([np.arange(0, r_inv.size(0)).tolist(),
                                     np.arange(0, r_inv.size(0)).tolist()],
                                     r_inv_values,
                                     size=(r_inv.size(0), r_inv.size(0)))
    # mx = r_mat_inv.mm(mx.float())
    mx_indices, mx_values = torch_sparse.spspmm(r_mat_inv._indices(), r_mat_inv._values().float(), 
                                                    mx._indices(), mx._values().float(),
                                                    r_mat_inv.size(0), r_mat_inv.size(1), mx.size(1),
                                                    coalesced=True)
    mx = torch.sparse_coo_tensor(indices=mx_indices, values=mx_values, size=(r_mat_inv.size(0), mx.size(1)))
    return mx

def one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, \
                            normalize):
    node_count = inputs.size(0)
    # n_per_proc = math.ceil(float(node_count) / size)
    n_per_proc = math.ceil(float(node_count) / (size / replication))

    am_partitions = None
    am_pbyp = None

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        am_pbyp, _ = split_coo(am_partitions[rank_c], node_count, n_per_proc, 0)
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
                                                        size=(n_per_proc, proc_node_count),
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

        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def oned_partition_mb(rank, size, batches, mb_count):
    # TODO: Include replicaiton at some point
    batch_partitions = torch.split(batches, math.ceil(float(mb_count / size)), dim=0)
    return batch_partitions[rank]

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
    # load and preprocess dataset
    device = torch.device('cuda')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)

    if args.dataset == "cora" or args.dataset == "reddit":
        if args.dataset == "cora":
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset == "reddit":
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
        n = 14249639
        num_features = 300
        num_classes = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        data.test_mask = torch.ones(n).long()
        adj_matrix = edge_index.t_()
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

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

    adj_matrix, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))

    partitioning = Partitioning.ONE5D

    row_groups, col_groups = get_proc_groups(rank, size, args.replication)

    rank_c = rank // args.replication
    rank_col = rank % args.replication
    if rank_c >= (size // args.replication):
        return

    print("start partitioning", flush=True)
    features_loc, g_loc, ampbyp = one5d_partition(rank, size, inputs, adj_matrix, data, \
                                                      inputs, num_classes, args.replication, \
                                                      args.normalize)
    print("end partitioning", flush=True)

    features_loc = features_loc.to(device)
    g_loc = g_loc.to(device)
    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)

    print("coalescing", flush=True)
    g_loc = g_loc.coalesce().t()
    print("normalizing", flush=True)
    g_loc = row_normalize(g_loc)
    print("done normalizing", flush=True)

    n_per_proc = math.ceil(float(g_loc.size(0)) / (size / args.replication))

    rank_train_mask = torch.split(data.train_mask, n_per_proc, dim=0)[rank_c]
    rank_test_mask = torch.split(data.test_mask, n_per_proc, dim=0)[rank_c]
    labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank_c]
    rank_train_nids = rank_train_mask.nonzero().squeeze()
    rank_test_nids = rank_test_mask.nonzero().squeeze()

    train_nid = data.train_mask.nonzero().squeeze()
    test_nid = data.test_mask.nonzero().squeeze()

    batches = torch.cuda.IntTensor(args.n_bulkmb, args.batch_size) # initially the minibatch, note row-major
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-gen-minibatch-vtxs")
    torch.manual_seed(0)
    vertex_perm = torch.randperm(train_nid.size(0))
    # Generate minibatch vertices
    for i in range(args.n_bulkmb):
        idx = vertex_perm[(i * args.batch_size):((i + 1) * args.batch_size)]
        batches[i,:] = train_nid[idx]
    torch.cuda.nvtx.range_pop()

    batches_loc = oned_partition_mb(rank, size, batches, args.n_bulkmb)

    torch.cuda.nvtx.range_push("nvtx-gen-sparse-batches")
    batches_indices_rows = torch.arange(batches_loc.size(0)).cuda()
    batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
    batches_indices_cols = batches_loc.view(-1)
    batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
    batches_values = torch.cuda.FloatTensor(batches_loc.size(1) * batches_loc.size(0)).fill_(1.0)
    batches_loc = torch.sparse_coo_tensor(batches_indices, batches_values, (batches_loc.size(0), g_loc.size(1)))
    g_loc = torch.pow(g_loc, 2)
    torch.cuda.nvtx.range_pop()

    print(f"rank: {rank} g_loc: {g_loc}")
    print(f"rank: {rank} batches_loc: {batches_loc}")
    # do it once before timing
    current_frontier, next_frontier, adj_matrices = ladies_sampler(g_loc, batches_loc, args.batch_size, \
                                                                        args.samp_num, args.n_bulkmb, \
                                                                        args.n_layers, args.n_darts, \
                                                                        rank, size, col_groups[0])

    print()
    torch.cuda.profiler.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("nvtx-sampler")
    current_frontier, next_frontier, adj_matrices_bulk = ladies_sampler(g_loc, batches_loc, args.batch_size, \
                                                                            args.samp_num, args.n_bulkmb, \
                                                                            args.n_layers, args.n_darts, \
                                                                            rank, size, col_groups[0])
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.cudart().cudaProfilerStop()

    # adj_matrices[i][j] --  mb i layer j
    rank_n_bulkmb = int(args.n_bulkmb / size)
    adj_matrices = [[None] * args.n_layers for x in range(rank_n_bulkmb)] 
    for i in range(args.n_layers):
        for j in range(rank_n_bulkmb):
            row_select_min = j * args.batch_size
            row_select_max = (j + 1) * args.batch_size
            
            sampled_indices = adj_matrices_bulk[i]._indices()
            sampled_values = adj_matrices_bulk[i]._values()

            sample_select_mask = (row_select_min <= sampled_indices[0,:]) & \
                                 (sampled_indices[0,:] < row_select_max)
            adj_matrix_sample_indices = sampled_indices[:, sample_select_mask]
            adj_matrix_sample_indices[0,:] -= row_select_min
            adj_matrix_sample_values = sampled_values[sample_select_mask]

            adj_matrix_sample = torch.sparse_coo_tensor(adj_matrix_sample_indices, adj_matrix_sample_values, \
                                            size=(args.batch_size, args.samp_num + args.batch_size))
            adj_matrices[j][i] = adj_matrix_sample

    # print(f"sample: {adj_matrices}")
    node_count = inputs.size(0)
    adj_matrix = adj_matrix.cuda()
    adj_matrix = torch.sparse_coo_tensor(adj_matrix, 
                                            torch.cuda.FloatTensor(adj_matrix.size(1)).fill_(1.0), 
                                            size=(node_count, node_count))
    adj_matrix = adj_matrix.coalesce().cuda()
    adj_matrix = row_normalize(adj_matrix)
    adj_matrix = torch.pow(adj_matrix, 2)
    return current_frontier, next_frontier, adj_matrices, adj_matrix # return here while testing sampling code

    # create GCN model
    torch.manual_seed(0)
    model = GCN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.n_layers,
                      rank,
                      size,
                      partitioning,
                      args.replication,
                      device,
                      row_groups=row_groups,
                      col_groups=col_groups)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.manual_seed(0)
    total_start = time.time()
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}", flush=True)
        if epoch == 1:
            total_start = time.time()
        model.train()

        # forward
        logits = model(g_loc, features_loc, ampbyp)
        loss = CAGF.cross_entropy(logits[rank_train_nids], labels_rank[rank_train_nids], train_nid.size(0), \
                                        num_classes, partitioning, rank_c, col_groups[0], \
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
    total_stop = time.time()
    print(f"total_time: {total_stop - total_start}")

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
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="number of vertices in minibatch")
    parser.add_argument("--samp-num", type=int, default=64,
                        help="number of vertices per layer of layer-wise minibatch")
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
    parser.add_argument('--n-bulkmb', default=1, type=int,
                            help='number of minibatches to sample in bulk')
    parser.add_argument('--n-darts', default=-1, type=int,
                            help='number of darts to throw per minibatch in LADIES sampling')
    args = parser.parse_args()
    print(args)

    main(args)
