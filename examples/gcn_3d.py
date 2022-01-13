import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch_geometric
from torch_geometric.datasets import Planetoid, Reddit
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
import cagnet.nn.functional as CAGF

import socket

class GCN(nn.Module):
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, node_count, rank, size, partitioning, device,
                    row_groups=None, col_groups=None, c_groups=None, transpose_group=None):
    super(GCN, self).__init__()
    self.layers = nn.ModuleList()
    self.node_count = node_count
    self.rank = rank
    self.size = size
    self.row_groups = row_groups
    self.col_groups = col_groups
    self.c_groups = c_groups
    self.transpose_group = transpose_group
    self.device = device
    self.partitioning = partitioning
    self.n_classes = n_classes
    self.timings = dict()

    proc_row = CAGF.proc_row_size(size, partitioning)
    proc_col = CAGF.proc_col_size(size, partitioning)
    proc_c = CAGF.proc_c_size(size, partitioning)

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid

    self.proc_row = proc_row
    self.proc_col = proc_col
    self.proc_c = proc_c
    self.rank_row = rank_row
    self.rank_col = rank_col
    self.rank_c = rank_c

    # input layer
    self.layers.append(GCNConv(in_feats, n_hidden, self.partitioning, self.device))
    # hidden layers
    for i in range(n_layers - 1):
        self.layers.append(GCNConv(n_hidden, n_hidden, self.partitioning, self.device))
    # output layer
    self.layers.append(GCNConv(n_hidden, n_classes, self.partitioning, self.device))

  def forward(self, graph, inputs, ampbyp=None):
    h = inputs
    for l, layer in enumerate(self.layers):
      h = layer(self, graph, h)

    return h

def get_proc_groups(rank, size):
    proc_row = CAGF.proc_row_size(size, Partitioning.THREED)
    proc_col = CAGF.proc_col_size(size, Partitioning.THREED)
    proc_c = CAGF.proc_c_size(size, Partitioning.THREED)

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid
    
    row_groups = []
    col_groups = []
    c_groups = []

    row_procs = []
    col_procs = []
    c_procs = []

    for i in range(proc_row):
        row_groups_c = []
        row_procs_c = []
        for j in range(proc_c):
            proc_start = i * proc_col * proc_c + j
            proc_end = (i + 1) * proc_col * proc_c + j
            row_groups_c.append(dist.new_group(list(range(proc_start, proc_end, proc_c))))
            row_procs_c.append(list(range(proc_start, proc_end, proc_c)))
        row_groups.append(row_groups_c)
        row_procs.append(row_procs_c)

    for i in range(proc_col):
        col_groups_c =[]
        col_procs_c =[]
        for j in range(proc_c):
            proc_start = i * proc_c + j
            proc_end = proc_row * proc_col * proc_c + i * proc_c + j
            col_groups_c.append(dist.new_group(list(range(proc_start, proc_end, proc_c * proc_col))))
            col_procs_c.append(list(range(proc_start, proc_end, proc_c * proc_col)))
        col_groups.append(col_groups_c)
        col_procs.append(col_procs_c)

    for i in range(0, size, proc_c):
        c_groups.append(dist.new_group(list(range(i, i + proc_c))))
        c_procs.append(list(range(i, i + proc_c)))

    rank_t = rank_col * proc_col * proc_c + rank_row * proc_c + rank_c

    transpose_groups = []
    transpose_group = None

    for i in range(proc_row):
        transpose_groups_row = []
        for j in range(proc_col):
            transpose_groups_col = []
            for k in range(proc_c):
                local_rank = i * proc_col * proc_c + j * proc_c + k
                local_rank_t = j * proc_col * proc_c + i * proc_c + k

                if local_rank < local_rank_t:
                    transpose_groups_col.append(dist.new_group([local_rank, local_rank_t]))
                else:
                    transpose_groups_col.append(None)
            transpose_groups_row.append(transpose_groups_col)
        transpose_groups.append(transpose_groups_row)
    
    if rank < rank_t:
        transpose_group = transpose_groups[rank_row][rank_col][rank_c]
    else:
        transpose_group = transpose_groups[rank_col][rank_row][rank_c]

    return row_groups, col_groups, c_groups, transpose_group

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

def twod_partition(rank, size, inputs, adj_matrix, normalize, device):
    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    node_count = inputs.size(0)
    proc_row = CAGF.proc_row_size(size, Partitioning.THREED)
    proc_col = CAGF.proc_col_size(size, Partitioning.THREED)
    proc_c = CAGF.proc_c_size(size, Partitioning.THREED)

    # n_per_proc = math.ceil(float(node_count) / proc_row)
    n_per_proc = node_count // proc_row

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    
    am_partitions = None
    am_pbyp = None

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1, size)

        proc_node_count = vtx_indices[rank_col + 1] - vtx_indices[rank_col]
        am_pbyp, _ = split_coo(am_partitions[rank_col], node_count, n_per_proc, 0, size)
        for i in range(len(am_pbyp)):
            if i == proc_row - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                    vtx_indices[rank_col], normalize)
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                    vtx_indices[rank_col], normalize)

        # input_rowparts = torch.split(inputs, math.ceil(float(inputs.size(0)) / proc_row), dim=0)
        inputs_per_row = inputs.size(0) // proc_row
        inputs_per_col = inputs.size(1) // proc_col
        chunks_per_row = []
        chunks_per_col = []
        for i in range(proc_row):
            if i == proc_row - 1:
                chunks_per_row.append(inputs.size(0) - inputs_per_row * (proc_row - 1))
            else:
                chunks_per_row.append(inputs_per_row)
        for i in range(proc_col):
            if i == proc_col - 1:
                chunks_per_col.append(inputs.size(1) - inputs_per_col * (proc_col - 1))
            else:
                chunks_per_col.append(inputs_per_col)

        # input_rowparts = torch.split(inputs, math.ceil(float(inputs.size(0)) / proc_row), dim=0)
        input_rowparts = torch.split(inputs, chunks_per_row, dim=0)
        input_partitions = []
        for i in input_rowparts:
            # input_partitions.append(torch.split(i, math.ceil(float(inputs.size(1)) / proc_col), 
            #                            dim=1))
            input_partitions.append(torch.split(i, chunks_per_col, dim=1))

        adj_matrix_loc = am_pbyp[rank_row]
        inputs_loc = input_partitions[rank_row][rank_col]

    return inputs_loc, adj_matrix_loc

def threed_partition_loc(rank, size, inputs, adj_matrix, height, width, device):
    proc_row = CAGF.proc_row_size(size, Partitioning.THREED)
    proc_col = CAGF.proc_col_size(size, Partitioning.THREED)
    proc_c = CAGF.proc_c_size(size, Partitioning.THREED)

    n_per_proc = width // proc_c

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid
    
    am_partitions, vtx_indices = split_coo(adj_matrix, width, n_per_proc, 1, size)

    for i in range(len(am_partitions)):
        if i == proc_c - 1:
            last_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], torch.ones(am_partitions[i].size(1)), 
                                                            size=(height, last_node_count),
                                                            requires_grad=False)

        else:
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], torch.ones(am_partitions[i].size(1)), 
                                                            size=(height, n_per_proc),
                                                            requires_grad=False)

    inputs_per_row = inputs.size(0) // proc_c
    chunks_per_row = []
    for i in range(proc_c):
        if i == proc_row - 1:
            chunks_per_row.append(inputs.size(0) - inputs_per_row * (proc_c - 1))
        else:
            chunks_per_row.append(inputs_per_row)

    input_rowparts = torch.split(inputs, chunks_per_row, dim=0)

    adj_matrix_loc = am_partitions[rank_c]
    inputs_loc = input_rowparts[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)

    return inputs_loc, adj_matrix_loc 


# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim, size):
    proc_row = CAGF.proc_row_size(size, Partitioning.THREED)
    proc_col = CAGF.proc_col_size(size, Partitioning.THREED)

    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices = vtx_indices[:proc_row]
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

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

    if args.normalize:
        adj_matrix, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))

    partitioning = Partitioning.THREED

    row_groups, col_groups, c_groups, transpose_group = get_proc_groups(rank, size)

    proc_row = CAGF.proc_row_size(size, Partitioning.THREED)
    proc_col = CAGF.proc_col_size(size, Partitioning.THREED)
    proc_c = CAGF.proc_c_size(size, Partitioning.THREED)

    rank_row = int((rank // proc_c) // proc_col) # i in process grid
    rank_col = int((rank // proc_c) % proc_col)  # j in process grid
    rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid
    
    print(f"Before partitioning...", flush=True)
    features_loc, g_loc = twod_partition(rank, size, inputs, adj_matrix, args.normalize, device)

    g_loc = g_loc.coalesce()

    features_loc, g_loc = threed_partition_loc(rank, size, features_loc, g_loc.indices(), 
                                                                g_loc.size(0), g_loc.size(1),
                                                                device)
    print(f"After partitioning...", flush=True)

    g_loc = g_loc.coalesce()
    features_loc = features_loc.to(device)
    g_loc = g_loc.to(device)

    # create GCN model
    torch.manual_seed(0)
    model = GCN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.n_layers,
                      inputs.size(0),
                      rank,
                      size,
                      partitioning,
                      device,
                      row_groups=row_groups,
                      col_groups=col_groups,
                      c_groups=c_groups,
                      transpose_group=transpose_group)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_per_proc = inputs.size(0) // proc_row

    rank_train_mask = torch.split(data.train_mask, inputs.size(0), dim=0)[rank_row * proc_c + rank_c]
    labels_rank = torch.split(data.y, inputs.size(0), dim=0)[rank_row * proc_c + rank_c]

    train_nid = data.train_mask.nonzero().squeeze()
    test_nid = data.test_mask.nonzero().squeeze()

    torch.manual_seed(0)
    total_start = time.time()
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}", flush=True)
        if epoch == 1:
            total_start = time.time()
        model.train()

        # forward
        logits = model(g_loc, features_loc)
        loss = CAGF.cross_entropy(logits, labels_rank[rank_train_mask], train_nid.size(0), \
                                        num_classes, partitioning, rank, [row_groups, col_groups, c_groups], \
                                        size, data.train_mask)

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
    parser.add_argument('--partitioning', default='THREED', type=str,
                            help='partitioning strategy to use')
    args = parser.parse_args()
    print(args)

    main(args)
