import os
import os.path as osp
import argparse

import math

import torch
import torch.distributed as dist

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI
from reddit import Reddit
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, dense_to_sparse, to_scipy_sparse_matrix
import torch_geometric.transforms as T

import torch.multiprocessing as mp

from torch.multiprocessing import Manager, Process

from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_add
import torch_sparse

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

import socket
import statistics
import time
import numpy as np

# comp_time = 0.0
# comm_time = 0.0
# scomp_time = 0.0
# dcomp_time = 0.0
# bcast_comm_time = 0.0
# bcast_words = 0
# op1_comm_time = 0.0
# op2_comm_time = 0.0
total_time = dict()
comp_time = dict()
comm_time = dict()
scomp_time = dict()
dcomp_time = dict()
bcast_comm_time = dict()
bcast_words = dict()
reduce_comm_time = dict()
op_comm_time = dict()
barrier_time = dict()

epochs = 0
graphname = ""
mid_layer = 0
timing = True
normalization = False
activations = False
accuracy = False
device = None
acc_per_rank = 0
run_count = 0
run = 0
replication = 0
download = False

def start_time(group, rank, subset=False, src=None):
    global barrier_time
    global run

    if not timing:
        return 0.0
    if group is not None:
        barrier_tstart = time.time()
        # dist.barrier(group)
        torch.cuda.synchronize(device=device)
        barrier_tstop = time.time()
        barrier_time[run][rank] += barrier_tstop - barrier_tstart
    tstart = 0.0
    if rank == 0:
        tstart = time.time()
    return tstart

def stop_time(group, rank, tstart):
    global barrier_time
    global run

    if not timing:
        return 0.0
    if group is not None:
        barrier_tstart = time.time()
        # dist.barrier(group)
        torch.cuda.synchronize(device=device)
        barrier_tstop = time.time()
        barrier_time[run][rank] += barrier_tstop - barrier_tstart
    tstop = 0.0
    if rank == 0:
        tstop = time.time()
    return tstop - tstart

def normalize(adj_matrix):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    d = torch.sum(adj_matrix, dim=1)
    d = torch.rsqrt(d)
    d = torch.diag(d)
    return torch.mm(d, torch.mm(adj_matrix, d))

def block_row(adj_matrix, am_partitions, inputs, weight, rank, size):
    n_per_proc = math.ceil(float(adj_matrix.size(1)) / size)
    # n_per_proc = int(adj_matrix.size(1) / size)
    # am_partitions = list(torch.split(adj_matrix, n_per_proc, dim=1))

    z_loc = torch.cuda.FloatTensor(n_per_proc, inputs.size(1)).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))
    
    inputs_recv = torch.zeros(inputs.size())

    part_id = rank % size

    z_loc += torch.mm(am_partitions[part_id].t(), inputs) 

    for i in range(1, size):
        part_id = (rank + i) % size

        inputs_recv = torch.zeros(am_partitions[part_id].size(0), inputs.size(1))

        src = (rank + 1) % size
        dst = rank - 1
        if dst < 0:
            dst = size - 1

        if rank == 0:
            dist.send(tensor=inputs, dst=dst)
            dist.recv(tensor=inputs_recv, src=src)
        else:
            dist.recv(tensor=inputs_recv, src=src)
            dist.send(tensor=inputs, dst=dst)
        
        inputs = inputs_recv.clone()

        # z_loc += torch.mm(am_partitions[part_id], inputs) 
        z_loc += torch.mm(am_partitions[part_id].t(), inputs) 


    # z_loc = torch.mm(z_loc, weight)
    return z_loc

def outer_product2(inputs, ag, rank, size, group):
    global comm_time
    global comp_time
    global dcomp_time
    global op_comm_time
    global run

    tstart_comp = start_time(group, rank)
    # (H^(l-1))^T * (A * G^l)
    grad_weight = torch.mm(inputs, ag)

    dur = stop_time(group, rank, tstart_comp)
    comp_time[run][rank] += dur
    dcomp_time[run][rank] += dur
    
    tstart_comm = start_time(group, rank)
    # reduction on grad_weight low-rank matrices
    dist.all_reduce(grad_weight, op=dist.reduce_op.SUM, group=group)

    dur = stop_time(group, rank, tstart_comm)
    comm_time[run][rank] += dur
    op_comm_time[run][rank] += dur

    return grad_weight

def broad_func(node_count, am_partitions, inputs, rank, size, row_groups, col_groups, group):
    global device
    global comm_time
    global comp_time
    global scomp_time
    global bcast_comm_time
    global bcast_words
    global reduce_comm_time
    global run
    global replication

    # n_per_proc = math.ceil(float(adj_matrix.size(1)) / size)
    n_per_proc = math.ceil(float(node_count) / (size / replication))

    # z_loc = torch.cuda.FloatTensor(adj_matrix.size(0), inputs.size(1), device=device).fill_(0)
    z_loc = torch.cuda.FloatTensor(am_partitions[0].size(0), inputs.size(1), device=device).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))

    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=device).fill_(0)
    # inputs_recv = torch.zeros(n_per_proc, inputs.size(1))

    rank_c = rank // replication
    rank_col = rank % replication

    stages = size // (replication ** 2)
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages

    for i in range(stages):
        # q = rank_c // (size // (replication ** 2)) * (size // (replication ** 2)) + i
        # = q * replication + rank_c // (size // (replication **2))
        q = (rank_col * (size // (replication ** 2)) + i) * replication + rank_col

        q_c = q // replication

        am_partid = rank_col * (size // replication ** 2) + i

        if q == rank:
            inputs_recv = inputs.clone()
        elif q_c == size // replication - 1:
            inputs_recv = torch.cuda.FloatTensor(am_partitions[am_partid].size(1), inputs.size(1), device=device).fill_(0)
            # inputs_recv = torch.zeros(list(am_partitions[i].t().size())[1], inputs.size(1))

        tstart_comm = start_time(col_groups[rank_col], rank)

        inputs_recv = inputs_recv.contiguous()
        bcast_words[run][rank] += inputs_recv.size(0) * inputs_recv.size(1)
        dist.broadcast(inputs_recv, src=q, group=col_groups[rank_col])

        dur = stop_time(col_groups[rank_col], rank, tstart_comm)

        comm_time[run][rank] += dur
        bcast_comm_time[run][rank] += dur

        tstart_comp = start_time(col_groups[rank_col], rank)

        spmm_gpu(am_partitions[am_partid].indices()[0].int(), am_partitions[am_partid].indices()[1].int(), 
                        am_partitions[am_partid].values(), am_partitions[am_partid].size(0), 
                        am_partitions[am_partid].size(1), inputs_recv, z_loc)

        dur = stop_time(col_groups[rank_col], rank, tstart_comp)
        comp_time[run][rank] += dur
        scomp_time[run][rank] += dur

    z_loc = z_loc.contiguous()

    tstart_comm = start_time(row_groups[rank_c], rank)
    dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=row_groups[rank_c])
    dur = stop_time(row_groups[rank_c], rank, tstart_comm)

    comm_time[run][rank] += dur
    reduce_comm_time[run][rank] += dur

    return z_loc

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions, rank, size, group, row_groups, col_groups, func):
        global comm_time
        global comp_time
        global dcomp_time
        global run

        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        # adj_matrix = adj_matrix.to_dense()
        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.rank = rank
        ctx.size = size
        ctx.group = group
        ctx.row_groups = row_groups
        ctx.col_groups = col_groups
        ctx.am_partitions = am_partitions

        ctx.func = func

        # z = block_row(adj_matrix.t(), am_partitions, inputs, weight, rank, size)
        z = broad_func(adj_matrix.size(0), am_partitions, inputs, rank, size, row_groups, col_groups, group)

        tstart_comp = start_time(row_groups[0], rank)

        z = torch.mm(z, weight)

        dur = stop_time(row_groups[0], rank, tstart_comp)
        comp_time[run][rank] += dur
        dcomp_time[run][rank] += dur

        z.requires_grad = True
        ctx.z = z

        if activations:
            if func is F.log_softmax:
                h = func(z, dim=1)
            elif func is F.relu:
                h = func(z)
            else:
                h = z

            return h
        else:
            return z

    @staticmethod
    def backward(ctx, grad_output):
        global comm_time
        global comp_time
        global dcomp_time
        global run

        inputs, weight, adj_matrix = ctx.saved_tensors
        rank = ctx.rank
        size = ctx.size
        group = ctx.group
        row_groups = ctx.row_groups
        col_groups = ctx.col_groups
        am_partitions = ctx.am_partitions

        func = ctx.func
        z = ctx.z

        rank_col = rank % replication

        if activations:
            with torch.set_grad_enabled(True):
                if func is F.log_softmax:
                    func_eval = func(z, dim=1)
                elif func is F.relu:
                    func_eval = func(z)
                else:
                    func_eval = z

            sigmap = torch.autograd.grad(outputs=func_eval, inputs=z, grad_outputs=grad_output)[0]
            grad_output = sigmap

        # First backprop equation
        # ag = outer_product(adj_matrix, grad_output, rank, size, group)
        ag = broad_func(adj_matrix.size(0), am_partitions, grad_output, rank, size, row_groups, col_groups, group)

        tstart_comp = start_time(group, rank)

        grad_input = torch.mm(ag, weight.t())

        dur = stop_time(group, rank, tstart_comp)
        comp_time[run][rank] += dur
        dcomp_time[run][rank] += dur

        # Second backprop equation (reuses the A * G^l computation)
        # grad_weight = outer_product2(inputs.t(), ag, rank, size, group)
        grad_weight = outer_product2(inputs.t(), ag, rank, size, col_groups[rank_col])

        return grad_input, grad_weight, None, None, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer, data, rank, size, group, row_groups, col_groups):

    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions, rank, size, group, row_groups, col_groups, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions, rank, size, group, row_groups, col_groups, F.log_softmax)

    optimizer.zero_grad()

    rank_c = rank // replication
    n_per_proc = int(math.ceil(float(node_count) / (size / replication)))
    rank_train_mask = torch.split(data.train_mask.bool(), n_per_proc, dim=0)[rank_c]
    datay_rank = torch.split(data.y, n_per_proc, dim=0)[rank_c]

    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
    # if datay_rank.size(0) > 0:
        loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        # loss = F.nll_loss(outputs, torch.max(datay_rank, 1)[1])
        loss.backward()
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size(), device=device).fill_(0)).sum()
        # fake_loss = (outputs * torch.zeros(outputs.size())).sum()
        fake_loss.backward()

    optimizer.step()

    return outputs

def test(outputs, data, vertex_count, rank):
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
    # logits, accs = outputs, []
    # datay_rank = torch.split(data.y, vertex_count)[rank]
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #     mask_rank = torch.split(mask, vertex_count)[rank]
    #     count = mask_rank.nonzero().size(0)
    #     if count > 0:
    #         pred = logits[mask_rank].max(1)[1]
    #         acc = pred.eq(datay_rank[mask_rank]).sum().item() / mask_rank.sum().item()
    #         # pred = logits[mask].max(1)[1]
    #         # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     else:
    #         acc = -1
    #     accs.append(acc)
    # return accs

def get_proc_groups(rank, size):
    global replication
    
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

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx):
    if not normalization:
        return adj_part

    # Scale each edge (u, v) by 1 / (sqrt(u) * sqrt(v))
    # indices = adj_part._indices()
    # values = adj_part._values()

    # deg_map = dict()
    # for i in range(adj_part._nnz()):
    #     u = indices[0][i] + row_vtx
    #     v = indices[1][i] + col_vtx

    #     if u.item() in deg_map:
    #         degu = deg_map[u.item()]
    #     else:
    #         degu = (adj_matrix[0] == u).sum().item()
    #         deg_map[u.item()] = degu

    #     if v.item() in deg_map:
    #         degv = deg_map[v.item()]
    #     else:
    #         degv = (adj_matrix[0] == v).sum().item()
    #         deg_map[v.item()] = degv

    #     values[i] = values[i] / (math.sqrt(degu) * math.sqrt(degv))
    
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

def oned_partition(rank, size, inputs, adj_matrix, data, features, classes, device):
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
                                                vtx_indices[rank_c])
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank_c])

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i])

        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def run(rank, size, inputs, adj_matrix, data, features, classes, device):
    global epochs
    global mid_layer
    global timing
    global run

    best_val_acc = test_acc = 0
    outputs = None

    # adj_matrix_loc = torch.rand(node_count, n_per_proc)
    # inputs_loc = torch.rand(n_per_proc, inputs.size(1))

    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size) 

    rank_c = rank // replication
    rank_col = rank % replication
    if rank_c >= (size // replication):
        return

    inputs_loc, adj_matrix_loc, am_pbyp = oned_partition(rank, size, inputs, adj_matrix, data, 
                                                                features, classes, device)

    inputs_loc = inputs_loc.to(device)
    adj_matrix_loc = adj_matrix_loc.to(device)
    for i in range(len(am_pbyp)):
        am_pbyp[i] = am_pbyp[i].t().coalesce().to(device)

    adj_matrix_loc.coalesce()
    dist.barrier(group)

    for i in range(run_count):
        run = i
        torch.manual_seed(0)
        weight1_nonleaf = torch.rand(features, mid_layer, requires_grad=True)
        weight1_nonleaf = weight1_nonleaf.to(device)
        weight1_nonleaf.retain_grad()

        weight2_nonleaf = torch.rand(mid_layer, classes, requires_grad=True)
        weight2_nonleaf = weight2_nonleaf.to(device)
        weight2_nonleaf.retain_grad()

        weight1 = Parameter(weight1_nonleaf)
        weight2 = Parameter(weight2_nonleaf)

        optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)

        total_time[i] = dict()
        comm_time[i] = dict()
        comp_time[i] = dict()
        scomp_time[i] = dict()
        dcomp_time[i] = dict()
        bcast_comm_time[i] = dict()
        bcast_words[i] = dict()
        reduce_comm_time[i] = dict()
        op_comm_time[i] = dict()
        barrier_time[i] = dict()

        total_time[i][rank] = 0.0
        comm_time[i][rank] = 0.0
        comp_time[i][rank] = 0.0
        scomp_time[i][rank] = 0.0
        dcomp_time[i][rank] = 0.0
        bcast_comm_time[i][rank] = 0.0
        bcast_words[i][rank] = 0
        reduce_comm_time[i][rank] = 0.0
        op_comm_time[i][rank] = 0.0
        barrier_time[i][rank] = 0.0

        # Do not time first epoch
        timing_on = timing == True
        timing = False

        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, data, 
                                    rank, size, group, row_groups, col_groups)
        if timing_on:
            timing = True

        dist.barrier(group)
        tstart = time.time()

        # for epoch in range(1, 201):
        print(f"Starting training... rank {rank} run {i}", flush=True)
        for epoch in range(1, epochs):
            outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, data, 
                                    rank, size, group, row_groups, col_groups)
            print("Epoch: {:03d}".format(epoch), flush=True)

        # dist.barrier(group)
        torch.cuda.synchronize(device=device)
        tstop = time.time()
        total_time[i][rank] = tstop - tstart

    # Get median runtime according to rank0 and print that run's breakdown
    dist.barrier(group)

    if rank == 0:
        total_times_r0 = [] 
        for i in range(run_count):
            total_times_r0.append(total_time[i][0])

        print(f"total_times_r0: {total_times_r0}")
        median_run_time = statistics.median(total_times_r0)
        median_idx = total_times_r0.index(median_run_time)
        median_idx = torch.cuda.LongTensor([median_idx])
    else:
        median_idx = torch.cuda.LongTensor([0])
        
    dist.barrier(group)
    # dist.broadcast(median_idx, src=0, group=group)        
    median_idx = median_idx.item()
    print(f"rank: {rank} median_run: {median_idx}")
    print(f"rank: {rank} total_time: {total_time[median_idx][rank]}")
    print(f"rank: {rank} comm_time: {comm_time[median_idx][rank]}")
    print(f"rank: {rank} comp_time: {comp_time[median_idx][rank]}")
    print(f"rank: {rank} scomp_time: {scomp_time[median_idx][rank]}")
    print(f"rank: {rank} dcomp_time: {dcomp_time[median_idx][rank]}")
    print(f"rank: {rank} bcast_comm_time: {bcast_comm_time[median_idx][rank]}")
    print(f"rank: {rank} bcast_words: {bcast_words[median_idx][rank]}")
    print(f"rank: {rank} reduce_comm_time: {reduce_comm_time[median_idx][rank]}")
    print(f"rank: {rank} op_comm_time: {op_comm_time[median_idx][rank]}")
    print(f"rank: {rank} barrier_time: {barrier_time[median_idx][rank]}")
    print(f"rank: {rank} {outputs}")
    
    
    if accuracy:
        # All-gather outputs to test accuracy
        output_parts = []
        # n_per_proc = math.ceil(float(inputs.size(0)) / size)
        n_per_proc = math.ceil(float(inputs.size(0)) / (size / replication))
        # print(f"rows: {am_pbyp[-1].size(0)} cols: {classes}", flush=True)
        for i in range(size // replication):
            output_parts.append(torch.cuda.FloatTensor(n_per_proc, classes, device=device).fill_(0))

        if outputs.size(0) != n_per_proc:
            pad_row = n_per_proc - outputs.size(0) 
            outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, classes, device=device)), dim=0)

        # dist.all_gather(output_parts, outputs)
        dist.all_gather(output_parts, outputs, group=col_groups[rank_col])
        # output_parts[rank] = outputs
        output_parts[rank_c] = outputs
        
        padding = inputs.size(0) - n_per_proc * ((size // replication) - 1)
        output_parts[(size // replication) - 1] = output_parts[(size // replication) - 1][:padding,:]

        outputs = torch.cat(output_parts, dim=0)

        train_acc, val_acc, tmp_test_acc = test(outputs, data, am_pbyp[0].size(1), rank)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

        print(log.format(900, train_acc, best_val_acc, test_acc))
    return outputs

def rank_to_devid(rank, acc_per_rank):
    return rank % acc_per_rank

def init_process(rank, size, inputs, adj_matrix, data, features, classes, device, outputs, fn):
    run_outputs = fn(rank, size, inputs, adj_matrix, data, features, classes, device)
    if outputs is not None:
        outputs[rank] = run_outputs.detach()

def main():
    global device
    global graphname

    print(socket.gethostname())
    seed = 0

    if not download:
        mp.set_start_method('spawn', force=True)
        outputs = None
        if "OMPI_COMM_WORLD_RANK" in os.environ.keys():
            os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

        # Initialize distributed environment with SLURM
        if "SLURM_PROCID" in os.environ.keys():
            os.environ["RANK"] = os.environ["SLURM_PROCID"]

        if "SLURM_NTASKS" in os.environ.keys():
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

        if "MASTER_ADDR" not in os.environ.keys():
            os.environ["MASTER_ADDR"] = "127.0.0.1"

        os.environ["MASTER_PORT"] = "1234"
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        size = dist.get_world_size()
        print("Processes: " + str(size))

        # device = torch.device('cpu')
        devid = rank_to_devid(rank, acc_per_rank)
        device = torch.device('cuda:{}'.format(devid))
        torch.cuda.set_device(device)
        curr_devid = torch.cuda.current_device()
        # print(f"curr_devid: {curr_devid}", flush=True)
        devcount = torch.cuda.device_count()

    if graphname == "Cora":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
        dataset = Planetoid(path, graphname, transform=T.NormalizeFeatures())
        data = dataset[0]
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif graphname == "Reddit":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
        dataset = Reddit(path, T.NormalizeFeatures())
        data = dataset[0]
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif graphname == 'Amazon':
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../data/Amazon/processed/data.pt")
        print(f"Done loading coo", flush=True)
        # edge_index = edge_index.t_()
        # n = 9430088
        n = 14249639
        # n = 14249640
        num_features = 300
        num_classes = 24
        # mid_layer = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        # edge_index = edge_index.to(device)
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        print(f"edge_index: {edge_index}", flush=True)
        data = data.to(device)
        # inputs = inputs.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif graphname == 'subgraph3':
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../data/subgraph3/processed/data.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542
        num_features = 128
        # mid_layer = 512
        # mid_layer = 64
        num_classes = 256
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

    if download:
        exit()

    if normalization:
        adj_matrix, _ = add_remaining_self_loops(edge_index, num_nodes=inputs.size(0))
    else:
        adj_matrix = edge_index


    init_process(rank, size, inputs, adj_matrix, data, num_features, num_classes, device, outputs, 
                    run)

    if outputs is not None:
        return outputs[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--accperrank", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--graphname", type=str)
    parser.add_argument("--timing", type=str)
    parser.add_argument("--midlayer", type=int)
    parser.add_argument("--runcount", type=int)
    parser.add_argument("--replication", type=int)
    parser.add_argument("--normalization", type=str)
    parser.add_argument("--activations", type=str)
    parser.add_argument("--accuracy", type=str)
    parser.add_argument("--download", type=bool)

    args = parser.parse_args()
    print(args)

    acc_per_rank = args.accperrank

    epochs = args.epochs
    graphname = args.graphname
    timing = args.timing == "True"
    mid_layer = args.midlayer
    run_count = args.runcount
    normalization = args.normalization == "True"
    activations = args.activations == "True"
    accuracy = args.accuracy == "True"
    replication = args.replication
    download = args.download

    if not download:
        if (epochs is None) or (graphname is None) or (timing is None) or (mid_layer is None) or (run_count is None):
            print(f"Error: missing argument {epochs} {graphname} {timing} {mid_layer} {run_count}")
            exit()

    print(f"Arguments: epochs: {epochs} graph: {graphname} timing: {timing} mid: {mid_layer} norm: {normalization} act: {activations} acc: {accuracy} runs: {run_count} rep: {replication}")
    
    print(main())
