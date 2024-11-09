import argparse
import examples.gcn_15d
import math
import numpy as np
import os
import os.path as osp
import socket
import torch
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.utils import add_remaining_self_loops

from cagnet.samplers.ladies import dist_spgemm1D
from cagnet.partitionings import Partitioning

def main(args):
    # Initialize inputs to spgemm (i.e. adjacency matrix, batch matrix)
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

    g_loc, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))
    g_loc = g_loc.to(device)

    print("coalescing", flush=True)
    g_loc = torch.sparse_coo_tensor(g_loc, torch.ones(g_loc.size(1)).cuda(), 
                                            size=(inputs.size(0), inputs.size(0)),
                                            requires_grad=False)
    g_loc = g_loc.coalesce().t()
    print("normalizing", flush=True)
    g_loc = examples.gcn_15d.row_normalize(g_loc)
    print("done normalizing", flush=True)

    partitioning = Partitioning.ONE5D

    row_groups, col_groups = examples.gcn_15d.get_proc_groups(rank, size, args.replication)

    rank_c = rank // args.replication
    rank_col = rank % args.replication
    if rank_c >= (size // args.replication):
        return

    batches = torch.cuda.IntTensor(args.n_bulkmb, args.batch_size) # initially the minibatch, note row-major
    node_count = g_loc.size(0)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-gen-minibatch-vtxs")
    torch.manual_seed(0)
    train_nid = data.train_mask.nonzero().squeeze()
    vertex_perm = torch.randperm(train_nid.size(0))
    # Generate minibatch vertices
    for i in range(args.n_bulkmb):
        idx = vertex_perm[(i * args.batch_size):((i + 1) * args.batch_size)]
        batches[i,:] = train_nid[idx]
    torch.cuda.nvtx.range_pop()

    batches_loc = batches
    torch.cuda.nvtx.range_push("nvtx-gen-sparse-batches")
    batches_indices_rows = torch.arange(batches_loc.size(0)).cuda()
    batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
    batches_indices_cols = batches_loc.view(-1)
    batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
    batches_values = torch.cuda.FloatTensor(batches_loc.size(1) * batches_loc.size(0)).fill_(1.0)
    batches_loc = torch.sparse_coo_tensor(batches_indices, batches_values, (batches_loc.size(0), g_loc.size(1)))
    g_loc = torch.pow(g_loc, 2)

    group = col_groups[0] # Assume replication factor == 1
    dist.barrier(group)

    # Run SpGEMM with only rank0, pre-partitioning
    if rank == 0:
        ans_indices, ans_values = dist_spgemm1D(batches_loc, g_loc, 0, 1, row_groups[0])

    dist.barrier(group)

    # Run partitioning
    g_loc, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))
    _, g_loc, _ = examples.gcn_15d.one5d_partition(rank, size, inputs, g_loc, data, \
                                                      inputs, num_classes, args.replication, \
                                                      args.normalize)
    g_loc = g_loc.to(device)
    print("coalescing", flush=True)
    g_loc = g_loc.coalesce().t()
    print("normalizing", flush=True)
    g_loc = examples.gcn_15d.row_normalize(g_loc)
    print("done normalizing", flush=True)

    batches_loc = examples.gcn_15d.oned_partition_mb(rank, size, batches, args.n_bulkmb)
    torch.cuda.nvtx.range_push("nvtx-gen-sparse-batches")
    batches_indices_rows = torch.arange(batches_loc.size(0)).cuda()
    batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
    batches_indices_cols = batches_loc.view(-1)
    batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
    batches_values = torch.cuda.FloatTensor(batches_loc.size(1) * batches_loc.size(0)).fill_(1.0)
    batches_loc = torch.sparse_coo_tensor(batches_indices, batches_values, (batches_loc.size(0), g_loc.size(1)))
    g_loc = torch.pow(g_loc, 2)

    # group = col_groups[0] # Assume replication factor == 1
    dist.barrier(group)

    # Run SpGEMM with only rank0, pre-partitioning
    test_indices, test_values = dist_spgemm1D(batches_loc, g_loc, rank, size, group)

    dist.barrier(group)

    if rank == 0:
        print(f"ans_indices: {ans_indices}")
        print(f"ans_values: {ans_values}")

    print(f"rank: {rank} test_indices: {test_indices}")
    print(f"rank: {rank} test_values: {test_values}")

    for i in range(1, size):
        if rank == i:
            nnz_count = torch.cuda.LongTensor([test_values.size(0)])
            dist.send(nnz_count, dst=0)
            dist.send(test_indices, dst=0)
            dist.send(test_values, dst=0)
        elif rank == 0:
            nnz_count = torch.cuda.LongTensor([test_values.size(0)])
            dist.recv(nnz_count, src=i)
            test_indices_recv = torch.cuda.LongTensor(2, nnz_count.item())
            dist.recv(test_indices_recv, src=i)
            test_values_recv = torch.cuda.FloatTensor(nnz_count.item())
            dist.recv(test_values_recv, src=i)

            chunk_size = math.ceil(float(args.n_bulkmb / size))
            test_indices_recv[0, :] += i * chunk_size
            test_indices = torch.cat((test_indices, test_indices_recv), dim=1)
            test_values = torch.cat((test_values, test_values_recv), dim=0)

    if rank == 0:
        print(f"final test_indices: {test_indices}") 
        print(f"final test_values: {test_values}") 
        print(f"finaleq test_indices: {(test_indices == ans_indices).all()}") 
        print(f"finaleq test_values: {(test_values == ans_values).all()}") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test distributed SpGEMM function')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="number of vertices in minibatch")
    parser.add_argument("--samp-num", type=int, default=64,
                        help="number of vertices per layer of layer-wise minibatch")
    parser.add_argument('--normalize', action="store_true",
                            help='normalize adjacency matrix')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--replication', default=1, type=int,
                            help='partitioning strategy to use')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Avaiable GPU ID')
    parser.add_argument('--sample-method', type=str, default='ladies',
                        help='Sampled Algorithms: ladies/fastgcn/full')
    parser.add_argument('--batch-num', type=int, default= 10,
                        help='Maximum Batch Number')
    parser.add_argument('--pool-num', type=int, default= 10,
                        help='Number of Pool')
    parser.add_argument('--n-bulkmb', type=int, default=1,
                        help='Number of minibatches to sample in bulk')
    parser.add_argument('--n-darts', type=int, default=10,
                        help='Number of darts to throw per minibatch in LADIES sampling')
    args = parser.parse_args()
    print(args)

    main(args)
