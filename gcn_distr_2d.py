import os
import os.path as osp
import argparse

import math

import torch
import torch_sparse
import torch.distributed as dist

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI
from reddit import Reddit
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import (
        add_remaining_self_loops, 
        to_dense_adj, 
        dense_to_sparse, 
        to_scipy_sparse_matrix
)
import torch_geometric.transforms as T

import torch.multiprocessing as mp

from torch.multiprocessing import Manager, Process

import statistics

from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_add

import socket
import time
import numpy as np

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

# comp_time = 0.0
# comm_time = 0.0
# summa_sparse_bcast1 = 0.0
# summa_sparse_bcast1_words = 0
# summa_sparse_bcast2_words = 0
# summa_sparse_bcast2 = 0.0
# summa_sparse_bcast2_fwd = 0.0
# summa_sparse_bcast2_bwd = 0.0
# summa_bcast1 = 0.0
# summa_bcast2 = 0.0
# summa_sparse_comp = 0.0
# summa_comp = 0.0
# summa_loc_bcast = 0.0
# fwd_time = 0.0
# bwd_time = 0.0
# transpose_time = 0.0
# grad_weight_time = 0.0
# loss_calc_time = 0.0
# summa_sparse_time = 0.0
# summa_time = 0.0
# summa_loc_time = 0.0
total_time = dict()
comp_time = dict()
comm_time = dict()
summa_sparse_bcast1 = dict()
summa_sparse_bcast1_words = dict()
summa_sparse_bcast2_words = dict()
summa_sparse_bcast2 = dict()
summa_sparse_bcast2_fwd = dict()
summa_sparse_bcast2_bwd = dict()
summa_bcast1 = dict()
summa_bcast2 = dict()
summa_sparse_comp = dict()
summa_comp = dict()
summa_loc_bcast = dict()
fwd_time = dict()
bwd_time = dict()
transpose_time = dict()
grad_weight_time = dict()
loss_calc_time = dict()
summa_sparse_time = dict()
summa_time = dict()
summa_loc_time = dict()

epochs = 0
graphname = ""
mid_layer = 0
timing = False
normalization = False
activations = False
accuracy = False
no_occur_val = 42.1234
run_count = 0
run = 0
download = False

def sync_and_sleep(rank, device):
    torch.cuda.synchronize(device=device)
    print(f"Sleeping rank {rank}", flush=True)
    time.sleep(20)
    print(f"Done sleeping rank {rank}", flush=True)

def normalize(adj_matrix):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    d = torch.sum(adj_matrix, dim=1)
    d = torch.rsqrt(d)
    d = torch.diag(d)
    return torch.mm(d, torch.mm(adj_matrix, d))

def start_time(group, rank):
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))
    if not timing:
        return 0.0
    if group is not None:
        # dist.barrier(group)
        torch.cuda.synchronize(device=device)
    tstart = 0.0
    if rank == 0:
        tstart = time.time()
    return tstart

def stop_time(group, rank, tstart):
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))
    if not timing:
        return 0.0
    if group is not None:
       # dist.barrier(group)
       torch.cuda.synchronize(device=device)
    tstop = 0.0
    if rank == 0:
        tstop = time.time()
    return tstop - tstart

def transpose(mat, row, col, height, width, size, acc_per_rank, transpose_group):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    rank = row * proc_col + col
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    rank_t  = col * proc_row + row

    if rank == rank_t:
        return mat.t()

    # height_recv = math.ceil(float(width) / proc_row)
    # width_recv  = math.ceil(float(height) / proc_col)
    height_recv = width // proc_row
    width_recv  = height // proc_col

    if row == proc_row - 1:
        # height_recv -= proc_row * height_recv - width
        height_recv = width - height_recv * (proc_row - 1)

    if col == proc_col - 1:
        # width_recv -= proc_col * width_recv - height
        width_recv = height - width_recv * (proc_col - 1)

    mat_recv = torch.cuda.FloatTensor(height_recv, width_recv, device=device)

    # if rank < rank_t:
    #     dist.send(tensor=mat.t().contiguous(), dst=rank_t)
    #     dist.recv(tensor=mat_recv, src=rank_t)
    # else:
    #     dist.recv(tensor=mat_recv, src=rank_t)
    #     dist.send(tensor=mat.t().contiguous(), dst=rank_t)

    # transpose_group = dist.new_group([rank, rank_t])

    mat_recvs = [mat.t().contiguous(), mat_recv]

    if rank < rank_t:
        dist.broadcast(mat_recvs[0], src=rank, group=transpose_group)
        dist.broadcast(mat_recvs[1], src=rank_t, group=transpose_group)
        # dist.broadcast_multigpu([mat_recvs[0]], src=rank, group=transpose_group)
        # dist.broadcast_multigpu([mat_recvs[1]], src=rank_t, group=transpose_group)
    else:
        dist.broadcast(mat_recvs[1], src=rank_t, group=transpose_group)
        dist.broadcast(mat_recvs[0], src=rank, group=transpose_group)
        # dist.broadcast_multigpu([mat_recvs[1]], src=rank_t, group=transpose_group)
        # dist.broadcast_multigpu([mat_recvs[0]], src=rank, group=transpose_group)

    return mat_recvs[1]

def summa(adj_matrix, inputs, rank, row, col, size, acc_per_rank, row_groups, col_groups, height, 
            middim, width):

    global comm_time
    global comp_time

    global summa_bcast1
    global summa_bcast2

    global summa_comp
    global summa_time
    global run

    # tstart_summa_time = start_time(row_groups[0], rank)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    # height_per_proc = math.ceil(float(height) / proc_row)
    # width_per_proc  = math.ceil(float(width) / proc_col)
    # # TODO: Not sure how to handle this w/o square grid
    # middim_per_proc = math.ceil(float(middim) / proc_row)
    height_per_proc = height // proc_row
    width_per_proc  = width // proc_col
    # TODO: Not sure how to handle this w/o square grid
    middim_per_proc = middim // proc_row
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    if row == proc_row - 1:
        # height_per_proc -= proc_row * height_per_proc - height
        height_per_proc = height - height_per_proc * (proc_row - 1)

    if col == proc_col - 1:
        # width_per_proc -= proc_col * width_per_proc - width
        width_per_proc = width - width_per_proc * (proc_col - 1)

    acol_tens = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=device)
    brow_tens = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=device)

    acol = acol_tens
    brow = brow_tens

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=device).fill_(0)

    for k in range(proc_col):

        row_src_rank = k + proc_col * row
        col_src_rank = k * proc_col + col

        if k == proc_col - 1:
            # middim_per_proc -= proc_col * middim_per_proc - middim
            middim_per_proc = middim - middim_per_proc * (proc_col - 1)
            # acol_tens = acol_tens[:,:middim_per_proc]
            # brow_tens = brow_tens[:middim_per_proc]
            acol_tens = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=device)
            brow_tens = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=device)

        if row_src_rank == rank:
            acol = adj_matrix
        else:
            acol = acol_tens
            # acol = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=device)
        
        tstart = start_time(row_groups[row], rank)

        acol = acol.contiguous()
        # dist.broadcast_multigpu([acol], row_src_rank, row_groups[row])
        dist.broadcast(acol, row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time[run][rank] += dur
        summa_bcast1[run][rank] += dur

        if col_src_rank == rank:
            brow = inputs
        else:
            brow = brow_tens
            # brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=device)

        tstart = start_time(col_groups[col], rank)

        brow = brow.contiguous()
        # dist.broadcast_multigpu([brow], col_src_rank, col_groups[col])
        dist.broadcast(brow, col_src_rank, col_groups[col])

        dur = stop_time(col_groups[col], rank, tstart)
        comm_time[run][rank] += dur
        summa_bcast2[run][rank] += dur

        # tstart = start_time(row_groups[0], rank)
        tstart = start_time(None, rank)

        z_loc += torch.mm(acol.float(), brow)

        # dur = stop_time(row_groups[0], rank, tstart)
        dur = stop_time(None, rank, tstart)
        comp_time[run][rank] += dur
        summa_comp[run][rank] += dur

    # summa_time += stop_time(row_groups[0], rank, tstart_summa_time)
    return z_loc

def summa_sparse(adj_matrix, inputs, rank, row, col, size, acc_per_rank, row_groups, col_groups, 
                    height, middim, width):

    global comm_time
    global comp_time

    global summa_sparse_bcast1
    global summa_sparse_bcast2

    global summa_sparse_bcast1_words
    global summa_sparse_bcast2_words

    global summa_sparse_comp
    global summa_sparse_time
    global run

    # tstart_summa_sparse_time = start_time(row_groups[0], rank)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    # height_per_proc = math.ceil(float(height) / proc_row)
    # width_per_proc  = math.ceil(float(width) / proc_col)

    # # TODO: Not sure how to handle this w/o square grid
    # middim_per_proc = math.ceil(float(middim) / proc_col)
    height_per_proc = height // proc_row
    width_per_proc  = width // proc_col

    # TODO: Not sure how to handle this w/o square grid
    middim_per_proc = middim // proc_col
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    if row == proc_row - 1:
        # height_per_proc -= proc_row * height_per_proc - height
        height_per_proc = height - height_per_proc * (proc_row - 1)

    if col == proc_col - 1:
        # width_per_proc -= proc_col * width_per_proc - width
        width_per_proc = width - width_per_proc * (proc_col - 1)

    # acol = torch.cuda.sparse.FloatTensor(height_per_proc, middim_per_proc, device=device)

    # brow = torch.FloatTensor(middim_per_proc, width_per_proc)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=device).fill_(0)

    for k in range(proc_col):

        row_src_rank = k + proc_col * row
        col_src_rank = k * proc_col + col

        if k == proc_col - 1:
            # middim_per_proc -= proc_col * middim_per_proc - middim
            middim_per_proc = middim - middim_per_proc * (proc_col - 1)

        if row_src_rank == rank:
            # acol = adj_matrix.clone()
            acol_indices_len = torch.cuda.LongTensor(
                                            [adj_matrix.indices().contiguous()[0].size(0)], 
                                            device=device)
            acol_values_len = torch.cuda.LongTensor([adj_matrix.values().contiguous().size(0)],
                                                    device=device)
        else:
            # acol = torch.sparse.FloatTensor(height_per_proc, middim_per_proc)
            acol_indices_len = torch.cuda.LongTensor([0], device=device)
            acol_values_len = torch.cuda.LongTensor([0], device=device)

        dist.broadcast(acol_indices_len, row_src_rank, row_groups[row])
        # dist.broadcast_multigpu([acol_indices_len], row_src_rank, row_groups[row])

        acol_indices_len = acol_indices_len.item() # nnz
        # acol_values_len = acol_values_len.item()
        acol_values_len = acol_indices_len

        if row_src_rank == rank:
            acol_indices = adj_matrix.indices().contiguous().long()
            acol_values = adj_matrix.values().contiguous().float()
        else:
            acol_indices = torch.cuda.LongTensor(2, acol_indices_len, device=device).fill_(0)
            acol_values = torch.cuda.FloatTensor(acol_values_len, device=device).fill_(0)
        

        acol = torch.cat((acol_indices.float(), acol_values.unsqueeze(0)), dim=0).contiguous()

        tstart = start_time(row_groups[row], rank)

        # dist.broadcast_multigpu([acol], row_src_rank, row_groups[row])
        dist.broadcast(acol, row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time[run][rank] += dur
        summa_sparse_bcast1[run][rank] += dur
        if rank == 0:
            summa_sparse_bcast1_words[run][rank] += 3 * acol_values_len

        acol_indices = acol[:2].long()
        acol_values = acol[2].squeeze(0)

        if row_src_rank == rank:
            acol = adj_matrix
        else:
            acol = sparse_coo_tensor_gpu(acol_indices, acol_values, 
                                            torch.Size([height_per_proc, middim_per_proc]))


        if col_src_rank == rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=device)

        brow = brow.contiguous()

        tstart = start_time(row_groups[0], rank)
        # tstart = start_time(col_groups[col], rank)

        # dist.broadcast_multigpu([brow], col_src_rank, col_groups[col])
        dist.broadcast(brow, col_src_rank, col_groups[col])

        dur = stop_time(row_groups[0], rank, tstart)
        # dur = stop_time(col_groups[col], rank, tstart)

        comm_time[run][rank] += dur
        summa_sparse_bcast2[run][rank] += dur
        if rank == 0:
            summa_sparse_bcast2_words[run][rank] += brow.size(0) * brow.size(1)

        # tstart = start_time(row_groups[0], rank)
        tstart = start_time(None, rank)

        spmm_gpu(acol_indices[0].int(), acol_indices[1].int(), acol_values, 
                        height_per_proc, middim_per_proc, brow, z_loc)

        # dur = stop_time(row_groups[0], rank, tstart)
        dur = stop_time(None, rank, tstart)
        # dur = stop_time(col_groups[col], rank, tstart)
        comp_time[run][rank] += dur
        summa_sparse_comp[run][rank] += dur

    # summa_sparse_time += stop_time(row_groups[0], rank, tstart_summa_sparse_time)
    return z_loc

def summa_loc(mata, matb, rank, row, col, size, acc_per_rank, row_groups, col_groups, 
                    height, middim, width):

    global comm_time
    global comp_time

    global summa_loc_bcast
    global summa_loc_time
    global run

    # tstart_summa_loc_time = start_time(row_groups[0], rank)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    # height_per_proc = math.ceil(float(height) / proc_row)
    # width_per_proc  = math.ceil(float(width) / proc_col)
    # # TODO: Not sure how to handle this w/o square grid
    # middim_per_proc = math.ceil(float(middim) / proc_row)
    height_per_proc = height // proc_row
    width_per_proc  = width // proc_col
    # TODO: Not sure how to handle this w/o square grid
    middim_per_proc = middim // proc_row
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    if row == proc_row - 1:
        # height_per_proc -= proc_row * height_per_proc - height
        height_per_proc = height - height_per_proc * (proc_row - 1)

    # if col == proc_col - 1:
    #     width_per_proc -= proc_col * width_per_proc - width

    width_per_proc = matb[rank].size(1)

    acol_tens = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=device)
    brow_tens = torch.FloatTensor(middim_per_proc, width_per_proc)

    acol = acol_tens
    brow = brow_tens

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=device).fill_(0)

    for k in range(proc_col):

        row_src_rank = k + proc_col * row
        col_src_rank = k * proc_col + col

        if k == proc_col - 1:
            middim_per_proc -= proc_col * middim_per_proc - middim

        if row_src_rank == rank:
            acol = mata
        else:
            acol = acol_tens
            acol = torch.cuda.FloatTensor(height_per_proc, matb[col_src_rank].size(0), 
                                            device=device)
        
        tstart = start_time(row_groups[row], rank)

        acol = acol.contiguous()
        # dist.broadcast_multigpu([acol], row_src_rank, row_groups[row])
        dist.broadcast(acol, row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time[run][rank] += dur
        summa_loc_bcast[run][rank] += dur

        # if col_src_rank == rank:
        #     brow = matb.clone()
        # else:
        #     brow = torch.FloatTensor(middim_per_proc, width_per_proc)

        # dist.broadcast(brow, col_src_rank, col_groups[col])

        brow = matb[col_src_rank]

        # tstart = start_time(row_groups[0], rank)
        tstart = start_time(None, rank)

        z_loc += torch.mm(acol, brow)

        # dur = stop_time(row_groups[0], rank, tstart)
        dur = stop_time(None, rank, tstart)
        comp_time[run][rank] += dur

    # summa_loc_time += stop_time(row_groups[0], rank, tstart_summa_loc_time)
    return z_loc

def get_proc_groups(rank, size, group):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    
    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
        
    row_groups = []
    col_groups = []

    for i in range(proc_row):
        # dist.barrier(group)
        row_groups.append(dist.new_group(list(range(i * proc_col, i * proc_col + proc_col))))

    # dist.barrier(group)
    for i in range(proc_col):
        # dist.barrier(group)
        col_groups.append(dist.new_group(list(range(i, size, proc_row))))

    return row_groups, col_groups

def dist_log_softmax(z, rank, size, acc_per_rank, group):
    torch.set_printoptions(edgeitems=4)
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))
    
    maxes = torch.max(z, dim=1, keepdim=True)[0]
    maxes_recv = []
    for i in range(proc_col):
        maxes_recv.append(torch.cuda.FloatTensor(maxes.size(), device=device))

    # dist.all_reduce(maxes, op=dist.reduce_op.MAX, group=group)
    dist.all_gather(maxes_recv, maxes, group=group)
    maxes_recv[rank_col] = maxes
    maxes = torch.max(torch.cat(maxes_recv, dim=1), dim=1, keepdim=True)[0]

    h = torch.exp(z - maxes)
    sm_sum = torch.sum(h, dim=1, keepdim=True)

    sm_sum_recv = []
    for i in range(proc_col):
        sm_sum_recv.append(torch.cuda.FloatTensor(sm_sum.size(), device=device))

    # dist.all_reduce(sm_sum, op=dist.reduce_op.SUM, group=group)
    dist.all_gather(sm_sum_recv, sm_sum, group=group)
    sm_sum_recv[rank_col] = sm_sum
    sm_sum = torch.sum(torch.cat(sm_sum_recv, dim=1), dim=1, keepdim=True)
    sm_sum = torch.log(sm_sum)
    h = z - maxes - sm_sum
    return h

def dist_log_softmax2(z, rank, size, width, acc_per_rank, group, grad_output):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    chunk_sizes_col = []
    width_per_col = width // proc_col

    for i in range(proc_col):
        if i == proc_col - 1:
            chunk_sizes_col.append(width - width_per_col * (proc_col - 1))
        else:
            chunk_sizes_col.append(width_per_col)

    width_per_proc = width - width_per_col * (proc_col - 1)
    if z.size(1) != width_per_proc:
        z = torch.cat((z, torch.cuda.FloatTensor(z.size(0), width_per_proc - z.size(1))), dim=1)

    z_recv = []
    for i in range(proc_col):
        z_recv.append(torch.cuda.FloatTensor(z.size()))

    dist.all_gather(z_recv, z, group=group)
    z_recv[rank_col] = z

    for i in range(proc_col - 1):
        pad_col = width // proc_col
        z_recv[i] = z_recv[i][:,:pad_col]

    z = torch.cat(z_recv, dim=1)

    if grad_output is not None:
        if grad_output.size(1) != width_per_proc:
            grad_output = torch.cat((grad_output, 
                                        torch.cuda.FloatTensor(grad_output.size(0), 
                                                        width_per_proc - grad_output.size(1))), 
                                        dim=1)

        grad_output_recv = []
        for i in range(proc_col):
            grad_output_recv.append(torch.cuda.FloatTensor(grad_output.size()))

        dist.all_gather(grad_output_recv, grad_output, group=group)
        grad_output_recv[rank_col] = grad_output

        for i in range(proc_col - 1):
            pad_col = width // proc_col
            grad_output_recv[i] = grad_output_recv[i][:,:pad_col]

        grad_output = torch.cat(grad_output_recv, dim=1)

    maxes = torch.max(z, dim=1, keepdim=True)[0]
    h = torch.exp(z - maxes)
    sm_sum = torch.sum(h, dim=1, keepdim=True)
    sm_sum = torch.log(sm_sum)

    h = z - maxes - sm_sum

    # if h.requires_grad:
    #     if rank_col == 0:
    #         sm_sigma = torch.autograd.grad(outputs=h, inputs=z,
    #                                             grad_outputs=grad_output)[0]
    #         print(f"rank: {rank} sm_sigma: {sm_sigma}", flush=True)
    #     else:
    #         sm_sigma = torch.autograd.grad(outputs=h, inputs=z,
    #                                             grad_outputs=grad_output)[0]
    #         print(f"rank: {rank} sm_sigma: {sm_sigma}", flush=True)

    # Only works for P = 4
    # if rank_col == 0:
    #     return h[:,:width_per_proc]
    # else:
    #     return h[:,width_per_proc:]
    return h, z, grad_output

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, node_count, adj_matrix, am_partitions, rank, size, 
                        acc_per_rank, group, row_groups, col_groups, transpose_group, func):
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        global summa_sparse_bcast2
        global summa_sparse_bcast2_fwd
        global fwd_time
        global grad_weight_time
        global run

        # tstart = start_time(row_groups[0], rank)

        proc_row = proc_row_size(size)
        proc_col = proc_col_size(size)
        
        rank_row = int(rank / proc_col)
        rank_col = rank % proc_col
        device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.node_count = node_count
        ctx.rank = rank
        ctx.size = size
        ctx.acc_per_rank = acc_per_rank
        ctx.group = group
        ctx.row_groups = row_groups
        ctx.col_groups = col_groups
        ctx.transpose_group = transpose_group

        ctx.func = func

        adj_matrix_t = adj_matrix # Only true for undirected graphs

        tmp_summa_sparse_bcast2 = summa_sparse_bcast2[run][rank]

        # TODO: will need to change height argument when n % sqrt(P) != 0 and non-square grid
        z = summa_sparse(adj_matrix_t, inputs, rank, rank_row, rank_col, size, acc_per_rank, 
                            row_groups, col_groups, node_count, node_count, weight.size(0))

        # tstart_grad_weight = start_time(row_groups[0], rank)
        chunk_sizes_row = []
        chunk_sizes_col = []
        weight_per_row = weight.size(0) // proc_row
        weight_per_col = weight.size(1) // proc_col
        for i in range(proc_row):
            if i == proc_row - 1:
                chunk_sizes_row.append(weight.size(0) - weight_per_row * (proc_row - 1))
            else:
                chunk_sizes_row.append(weight_per_row)

        for i in range(proc_col):
            if i == proc_col - 1:
                chunk_sizes_col.append(weight.size(1) - weight_per_col * (proc_col - 1))
            else:
                chunk_sizes_col.append(weight_per_col)

        # weight_rows = torch.split(weight, math.ceil(float(weight.size(0)) / proc_row), dim=0)
        weight_rows = torch.split(weight, chunk_sizes_row, dim=0)
        weight_parts = []
        for i in weight_rows:
            # weight_cols = torch.split(i, math.ceil(float(weight.size(1)) / proc_col), dim=1)
            weight_cols = torch.split(i, chunk_sizes_col, dim=1)
            weight_parts.extend(weight_cols)
        # grad_weight_time += stop_time(row_groups[0], rank, tstart_grad_weight)

        # z = torch.mm(z, weight)
        z = summa_loc(z, weight_parts, rank, rank_row, rank_col, size, acc_per_rank, row_groups, 
                        col_groups, node_count, weight.size(0), weight.size(1))

        z.requires_grad = True
        ctx.z = z

        summa_sparse_bcast2_fwd[run][rank] += summa_sparse_bcast2[run][rank] - tmp_summa_sparse_bcast2

        if activations:
            if func is F.log_softmax:
                h = dist_log_softmax(z, rank, size, acc_per_rank, row_groups[rank_row])
            elif func is F.relu:
                h = func(z)
            else:
                h = z
            return h
        else:
            return z

        # dur = stop_time(row_groups[0], rank, tstart)
        # fwd_time += dur

        # return z

    @staticmethod
    def backward(ctx, grad_output):
        global summa_sparse_bcast2
        global summa_sparse_bcast2_bwd
        global bwd_time
        global transpose_time
        global grad_weight_time
        global run

        inputs, weight, adj_matrix = ctx.saved_tensors
        rank = ctx.rank
        size = ctx.size
        acc_per_rank = ctx.acc_per_rank
        group = ctx.group
        row_groups = ctx.row_groups
        col_groups = ctx.col_groups
        transpose_group = ctx.transpose_group
        node_count = ctx.node_count

        func = ctx.func
        z = ctx.z

        proc_row = proc_row_size(size)
        proc_col = proc_col_size(size)

        rank_row = int(rank / proc_col)
        rank_col = rank % proc_col
        device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

        # tstart = start_time(row_groups[0], rank)
            
        if activations:
            with torch.set_grad_enabled(True):
                if func is F.log_softmax:
                    # func_eval = dist_log_softmax2(z, rank, size, acc_per_rank, row_groups[rank_row])
                    func_eval, z_gathered, go_gathered = dist_log_softmax2(z, rank, size, 
                                                                    weight.size(1), 
                                                                    acc_per_rank, 
                                                                    row_groups[rank_row], grad_output)
                    width = z_gathered.size(1)

                    sigmap = torch.autograd.grad(outputs=func_eval, inputs=z_gathered,
                                                    grad_outputs=go_gathered)[0]

                    chunk_sizes_col = []
                    sigmap_per_col = width // proc_col

                    for i in range(proc_col):
                        if i == proc_col - 1:
                            chunk_sizes_col.append(width - sigmap_per_col * (proc_col - 1))
                        else:
                            chunk_sizes_col.append(sigmap_per_col)

                    grad_output = sigmap.split(chunk_sizes_col, dim=1)[rank_col]
                    del z_gathered
                    del go_gathered
                elif func is F.relu:
                    func_eval = func(z)
                    sigmap = torch.autograd.grad(outputs=func_eval, inputs=z,grad_outputs=grad_output)[0]
                    grad_output = sigmap
                else:
                    func_eval = z
                    sigmap = torch.autograd.grad(outputs=func_eval, inputs=z,grad_outputs=grad_output)[0]
                    grad_output = sigmap


        tmp_summa_sparse_bcast2 = summa_sparse_bcast2[run][rank]

        # First backprop equation
        # TODO: will need to change height argument when n % sqrt(P) != 0 and non-square grid
        ag = summa_sparse(adj_matrix, grad_output, rank, rank_row, rank_col, size, acc_per_rank, 
                            row_groups, col_groups, node_count, node_count, weight.t().size(0))

        # tstart_grad_weight = start_time(row_groups[0], rank)
        chunk_sizes_row = []
        chunk_sizes_col = []
        weight_per_row = weight.t().size(0) // proc_row
        weight_per_col = weight.t().size(1) // proc_col
        for i in range(proc_row):
            if i == proc_row - 1:
                chunk_sizes_row.append(weight.t().size(0) - weight_per_row * (proc_row - 1))
            else:
                chunk_sizes_row.append(weight_per_row)

        for i in range(proc_col):
            if i == proc_col - 1:
                chunk_sizes_col.append(weight.t().size(1) - weight_per_col * (proc_col - 1))
            else:
                chunk_sizes_col.append(weight_per_col)
        # weight_rows = torch.split(weight.t(), math.ceil(float(weight.t().size(0)) / proc_row), 
        weight_rows = torch.split(weight.t(), chunk_sizes_row, dim=0)

        weight_parts = []
        for i in weight_rows:
            # weight_cols = torch.split(i, math.ceil(float(weight.t().size(1)) / proc_col), dim=1)
            weight_cols = torch.split(i, chunk_sizes_col, dim=1)
            weight_parts.extend(weight_cols)

        # grad_input = torch.mm(ag, weight.t())
        grad_input = summa_loc(ag, weight_parts, rank, rank_row, rank_col, size, acc_per_rank, 
                                    row_groups, col_groups, node_count, weight.t().size(0), 
                                    weight.t().size(1))
        # grad_weight_time += stop_time(row_groups[0], rank, tstart_grad_weight)

        # Second backprop equation (reuses the A * G^l computation)
        # col_groups twice because of transpose
        # TODO: will need to change height argument when n % sqrt(P) != 0 and non-square grid

        # tstart_transpose = start_time(row_groups[0], rank)
        tstart_transpose = start_time(transpose_group, rank)
        inputs_t = transpose(inputs, rank_row, rank_col, node_count, weight.size(0), size,
                                acc_per_rank, transpose_group)
        # transpose_time[run][rank] += stop_time(row_groups[0], rank, tstart_transpose)
        transpose_time[run][rank] += stop_time(transpose_group, rank, tstart_transpose)

        grad_weight = summa(inputs_t, ag, rank, rank_row, rank_col, size, acc_per_rank, row_groups,
                                col_groups, weight.size(0), node_count, weight.size(1))

        # tstart_grad_weight = start_time(row_groups[0], rank)
        # Collect grad_weight's across processes
        grad_weight_recv = []
        max_row_chunk = max(chunk_sizes_col) #transpose
        max_col_chunk = max(chunk_sizes_row)
        for i in range(size):
            grad_weight_recv.append(torch.cuda.FloatTensor(
                                                max_row_chunk,
                                                max_col_chunk,
                                                device=device))

        # pad_row = math.ceil(float(weight.size(0)) / proc_row) - grad_weight.size(0)
        # pad_col = math.ceil(float(weight.size(1)) / proc_col) - grad_weight.size(1)
        pad_row = max_row_chunk - grad_weight.size(0)
        pad_col = max_col_chunk - grad_weight.size(1)

        # TODO: make this part less hacky
        grad_weight = torch.cat((grad_weight, 
                        torch.cuda.FloatTensor(pad_row, grad_weight.size(1), device=device).fill_(no_occur_val)), 
                        dim=0) 
        grad_weight = torch.cat((grad_weight, 
                        torch.cuda.FloatTensor(grad_weight.size(0), pad_col, device=device).fill_(no_occur_val)), 
                        dim=1) 

        dist.all_gather(grad_weight_recv, grad_weight)
        # dist.all_gather_multigpu([grad_weight_recv], [grad_weight])

        # for i in range(size):
        #     if rank == i:
        #         grad_weight_recv[i] = grad_weight
        #     dist.broadcast(grad_weight_recv[i], i, group)
        # grad_weight_recv[0] = grad_weight

        for i in range(len(grad_weight_recv)):
            grad_weight_recv[i] = grad_weight_recv[i][(grad_weight_recv[i][:, 0] != no_occur_val)
                                                                .nonzero().squeeze(1)]

            grad_weight_recv_t = grad_weight_recv[i].t()
            grad_weight_recv_t = grad_weight_recv_t[(grad_weight_recv_t[:, 0] != no_occur_val)
                                                                .nonzero().squeeze(1)]

            grad_weight_recv[i] = grad_weight_recv_t.t()
        
        grad_weight_fin = torch.cuda.FloatTensor(device=device)
        for i in range(proc_row):
            grad_weight_row = torch.cuda.FloatTensor(device=device)
            for j in range(proc_col):
                rank_wt = i * proc_row + j
                grad_weight_row = torch.cat((grad_weight_row, grad_weight_recv[rank_wt]), dim=1)
            grad_weight_fin = torch.cat((grad_weight_fin, grad_weight_row), dim=0)

        summa_sparse_bcast2_bwd[run][rank] += summa_sparse_bcast2[run][rank] - tmp_summa_sparse_bcast2

        # dur = stop_time(row_groups[0], rank, tstart)
        # bwd_time += dur

        # grad_weight_time += stop_time(row_groups[0], rank, tstart_grad_weight)

        return grad_input, grad_weight_fin, None, None, None, None, None, None, None, None, None, None, None

def train(inputs, weight1, weight2, node_count, adj_matrix, am_partitions, optimizer, data, rank, 
                size, acc_per_rank, group, row_groups, col_groups, transpose_group):

    global loss_calc_time
    global run

    outputs = GCNFunc.apply(inputs, weight1, node_count, adj_matrix, am_partitions, rank, size, 
                                    acc_per_rank, group, row_groups, col_groups, transpose_group, 
                                    F.relu)

    outputs = GCNFunc.apply(outputs, weight2, node_count, adj_matrix, am_partitions, rank, size, 
                                    acc_per_rank, group, row_groups, col_groups, transpose_group, 
                                    F.log_softmax)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    device = torch.device('cuda:{}'.format(rank_to_devid(rank, acc_per_rank)))

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask, outputs.size(0), dim=0)[rank_row]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[rank_row]

    total_classes = weight2.size(1)
    # class_per_rank = math.ceil(float(total_classes) / proc_col)
    class_per_rank = total_classes // proc_col

    min_class = rank_col * class_per_rank
    max_class = min((rank_col + 1) * class_per_rank, total_classes)
    if rank_col == proc_col - 1:
        max_classes = total_classes


    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
    # if datay_rank.size(0) > 0:
        # datay_ids = datay_rank[rank_train_mask].long().view(-1, 1)
        # tstart_loss_calc = start_time(row_groups[0], rank)

        datay_ids = datay_rank[rank_train_mask].long()

        filtered_indices = torch.mul(datay_ids >= min_class, datay_ids < max_class).float()
        indices = torch.nonzero(filtered_indices * torch.cuda.FloatTensor(datay_ids.size(), device=device).fill_(1)).squeeze()

        datay_ids = datay_rank[rank_train_mask].long().view(-1, 1)
        datay_ids = datay_ids.index_select(0, indices)
        datay_ids -= min_class
        outputs_ids = outputs.index_select(0, indices)

        # classes = torch.gather(outputs[rank_train_mask], 1, datay_ids)
        classes = torch.gather(outputs_ids, 1, datay_ids)
        loss_calc = torch.sum(classes)
        loss_calc_tens = torch.Tensor([loss_calc.item()])

        rank_row_src = rank_row * proc_col

        # dist.reduce_multigpu([loss_calc], dst=rank_row_src, op=dist.reduce_op.SUM, group=row_groups[rank_row])
        # dist.broadcast_multigpu([loss_calc], src=rank_row_src, group=row_groups[rank_row]) 
        dist.reduce(loss_calc, dst=rank_row_src, op=dist.reduce_op.SUM, group=row_groups[rank_row])
        dist.broadcast(loss_calc, src=rank_row_src, group=row_groups[rank_row]) 

        vertex_train_count = (data.train_mask.size(0) - (data.train_mask == 0).sum(dim=0))
        loss_calc = -loss_calc / vertex_train_count

        # loss_calc_time[run][rank] += stop_time(row_groups[0], rank, tstart_loss_calc)

        loss_calc.backward()
        # print("loss_calc: " + str(loss_calc), flush=True)
        # loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        # loss.backward()
        # print("loss: " + str(loss), flush=True)
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

    if len(accs) != 3:
        accs = accs + [0] * (3 - len(accs))

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


# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim, size):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

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

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx):
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

def proc_row_size(size):
    return math.floor(math.sqrt(size))

def proc_col_size(size):
    return math.floor(math.sqrt(size))

def twod_partition(rank, size, inputs, adj_matrix, data, features, classes, device):
    node_count = inputs.size(0)
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    # n_per_proc = math.ceil(float(node_count) / proc_row)
    n_per_proc = node_count // proc_row

    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    
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

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                    vtx_indices[rank_col])
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                    vtx_indices[rank_col])

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

    print(adj_matrix_loc.size(), flush=True)
    print(inputs_loc.size(), flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def rank_to_devid(rank, acc_per_rank):
    return rank % acc_per_rank

def run(rank, size, inputs, adj_matrix, data, features, mid_layer, classes, device, acc_per_rank):
    global comm_time
    global comp_time
    global epochs
    global timing
    global run

    best_val_acc = test_acc = 0
    outputs = None

    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size, group)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    rank_t  = rank_col * proc_row + rank_row

    if rank_row >= proc_row or rank_col >= proc_col:
        return

    transpose_groups = []

    for i in range(proc_row):
        transpose_groups_row = []
        for j in range(proc_col):
            local_rank = i * proc_col + j
            local_rank_t = j * proc_row + i
            if local_rank < local_rank_t:
                transpose_groups_row.append(dist.new_group([local_rank, local_rank_t]))
            else:
                transpose_groups_row.append(None)
        transpose_groups.append(transpose_groups_row)
    
    if rank < rank_t:
        transpose_group = transpose_groups[rank_row][rank_col]
    else:
        transpose_group = transpose_groups[rank_col][rank_row]

    # adj_matrix_loc = torch.rand(node_count, n_per_proc)
    # inputs_loc = torch.rand(n_per_proc, inputs.size(1))

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

        inputs_loc, adj_matrix_loc, _ = twod_partition(rank, size, inputs, adj_matrix, data, features,
                                                            classes, device)

        adj_matrix_loc = adj_matrix_loc.coalesce()

        inputs_loc = inputs_loc.to(device)
        adj_matrix_loc = adj_matrix_loc.to(device)

        print(f"rank: {rank} adj_matrix_loc.nnz: {adj_matrix_loc._nnz()}")

        total_time[i] = dict()
        comp_time[i] = dict()
        comm_time[i] = dict()
        summa_sparse_bcast1[i] = dict()
        summa_sparse_bcast1_words[i] = dict()
        summa_sparse_bcast2_words[i] = dict()
        summa_sparse_bcast2[i] = dict()
        summa_sparse_bcast2_fwd[i] = dict()
        summa_sparse_bcast2_bwd[i] = dict()
        summa_bcast1[i] = dict()
        summa_bcast2[i] = dict()
        summa_sparse_comp[i] = dict()
        summa_comp[i] = dict()
        summa_loc_bcast[i] = dict()
        fwd_time[i] = dict()
        bwd_time[i] = dict()
        transpose_time[i] = dict()
        grad_weight_time[i] = dict()
        loss_calc_time[i] = dict()
        summa_sparse_time[i] = dict()
        summa_time[i] = dict()
        summa_loc_time[i] = dict()

        total_time[i][rank] = 0.0
        comp_time[i][rank] = 0.0
        comm_time[i][rank] = 0.0
        summa_sparse_bcast1[i][rank] = 0.0
        summa_sparse_bcast1_words[i][rank] = 0.0
        summa_sparse_bcast2_words[i][rank] = 0.0
        summa_sparse_bcast2[i][rank] = 0.0
        summa_sparse_bcast2_fwd[i][rank] = 0.0
        summa_sparse_bcast2_bwd[i][rank] = 0.0
        summa_bcast1[i][rank] = 0.0
        summa_bcast2[i][rank] = 0.0
        summa_sparse_comp[i][rank] = 0.0
        summa_comp[i][rank] = 0.0
        summa_loc_bcast[i][rank] = 0.0
        fwd_time[i][rank] = 0.0
        bwd_time[i][rank] = 0.0
        transpose_time[i][rank] = 0.0
        grad_weight_time[i][rank] = 0.0
        loss_calc_time[i][rank] = 0.0
        summa_sparse_time[i][rank] = 0.0
        summa_time[i][rank] = 0.0
        summa_loc_time[i][rank] = 0.0

        # Do not time first epoch
        # timing_on = timing == True
        # timing = False
        # outputs = train(inputs_loc, weight1, weight2, inputs.size(0), adj_matrix_loc, None, 
        #                         optimizer, data, rank, size, acc_per_rank, group, row_groups, 
        #                         col_groups, transpose_group)
        # if timing_on:
        #     timing = True

        # # tstart = start_time(group, rank)
        dist.barrier(group)
        tstart = time.time()

        print(f"Starting training... rank {rank} run {i}", flush=True)
        for epoch in range(0, epochs):
            outputs = train(inputs_loc, weight1, weight2, inputs.size(0), adj_matrix_loc, None, 
                                    optimizer, data, rank, size, acc_per_rank, group, row_groups, 
                                    col_groups, transpose_group)
            print("Epoch: {:03d}".format(epoch), flush=True)

        # dur = stop_time(group, rank, tstart)
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

    # dist.broadcast(median_idx, src=0, group=group)        
    median_idx = median_idx.item()
    print(f"rank: {rank} median_idx: {median_idx}")
    print(f"rank: {rank} Time: {total_time[median_idx][rank]}")
    print(f"rank: {rank} comm_time: {comm_time[median_idx][rank]}")
    print(f"rank: {rank} comp_time: {comp_time[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_comp: {summa_sparse_comp[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_bcast1: {summa_sparse_bcast1[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_bcast1_words: {summa_sparse_bcast1_words[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_bcast2: {summa_sparse_bcast2[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_bcast2_fwd: {summa_sparse_bcast2_fwd[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_bcast2_bwd: {summa_sparse_bcast2_bwd[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_bcast2_words: {summa_sparse_bcast2_words[median_idx][rank]}")
    print(f"rank: {rank} summa_comp: {summa_comp[median_idx][rank]}")
    print(f"rank: {rank} summa_bcast1: {summa_bcast1[median_idx][rank]}")
    print(f"rank: {rank} summa_bcast2: {summa_bcast2[median_idx][rank]}")
    print(f"rank: {rank} summa_loc_bcast: {summa_loc_bcast[median_idx][rank]}")
    print(f"rank: {rank} transpose_time: {transpose_time[median_idx][rank]}")
    print(f"rank: {rank} grad_weight_time: {grad_weight_time[median_idx][rank]}")
    print(f"rank: {rank} loss_calc_time: {loss_calc_time[median_idx][rank]}")
    print(f"rank: {rank} summa_sparse_time: {summa_sparse_time[median_idx][rank]}")
    print(f"rank: {rank} summa_time: {summa_time[median_idx][rank]}")
    print(f"rank: {rank} summa_loc_time: {summa_loc_time[median_idx][rank]}")
    print(f"rank: {rank} {outputs}")
    
    # All-gather outputs to test accuracy
    if accuracy:
        # All-gather across process row
        output_parts_row = []
        width_per_proc = classes // proc_col
        for i in range(proc_col):
            output_parts_row.append(torch.cuda.FloatTensor(outputs.size(0), classes - width_per_proc * (proc_col - 1)))

        if outputs.size(1) != classes - width_per_proc * (proc_col - 1):
            pad_col = (classes - width_per_proc * (proc_col - 1)) - outputs.size(1)
            outputs = torch.cat((outputs, torch.cuda.FloatTensor(outputs.size(0), pad_col, device=device)), dim=1)

        dist.all_gather(output_parts_row, outputs, group=row_groups[rank_row])
        for i in range(proc_col - 1):
            output_parts_row[i] = output_parts_row[i][:,:width_per_proc]

        outputs_row = torch.cat(output_parts_row, dim=1)

        # All-gather across process col
        output_parts_col = []
        height_per_proc = inputs.size(0) // proc_row
        for i in range(proc_row):
            output_parts_col.append(torch.cuda.FloatTensor(inputs.size(0) - height_per_proc * (proc_row - 1), classes))

        if outputs_row.size(0) != inputs.size(0) - height_per_proc * (proc_row - 1):
            pad_row = (inputs.size(0) - height_per_proc * (proc_col - 1)) - outputs_row.size(0)
            outputs_row = torch.cat((outputs_row, torch.cuda.FloatTensor(pad_row, classes, device=device)), dim=0)

        dist.all_gather(output_parts_col, outputs_row, group=col_groups[rank_col])
        for i in range(proc_row - 1):
            output_parts_col[i] = output_parts_col[i][:height_per_proc,:]

        outputs = torch.cat(output_parts_col, dim=0)

        train_acc, val_acc, tmp_test_acc = test(outputs, data, inputs.size(0), rank)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

        print(log.format(200, train_acc, best_val_acc, test_acc))
    return outputs

def init_process(rank, size, inputs, adj_matrix, data, features, mid_layer, classes, device, 
                    outputs, acc_per_rank, fn):

    run_outputs = fn(rank, size, inputs, adj_matrix, data, features, mid_layer, classes, device, 
                            acc_per_rank)
    if outputs is not None:
        outputs[rank] = run_outputs.detach()

def main():
    # graphname = 'Reddit'
    global graphname
    global mid_layer

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)

    # mid_layer = 16
    if graphname == 'Cora':
        dataset = Planetoid(path, graphname, transform=T.NormalizeFeatures())
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif graphname == 'Reddit':
        dataset = Reddit(path, T.NormalizeFeatures())
        data = dataset[0]
        num_features = dataset.num_features
        # num_classes = dataset.num_classes + 9
        num_classes = dataset.num_classes
        print(f"before edge_index: {data.edge_index.size()}")
    elif graphname == 'Amazon':
        # edge_index = torch.load(path + "/processed/amazon_graph.pt")
        # edge_index = torch.load("/gpfs/alpine/bif115/scratch/alokt/Amazon/processed/amazon_graph_random.pt")
        # edge_index = torch.load("/gpfs/alpine/bif115/scratch/alokt/Amazon/processed/amazon_large_randomized.pt")
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../data/Amazon/processed/data.pt")
        print(f"Done loading coo", flush=True)
        # edge_index = edge_index.t_()
        # n = 9430086
        # n = 9430088
        n = 14249639
        num_features = 300
        num_classes = 24
        # mid_layer = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1)
        data.train_mask = torch.ones(n).long()
        print(f"before edge_index: {edge_index.size()}")
    elif graphname == 'subgraph5':
        path = "/gpfs/alpine/bif115/scratch/alokt/HipMCL/"
        edge_index = torch.load(path + "/processed/subgraph5_graph.pt")
        n = 2186385
        num_features = 128
        # mid_layer = 64
        num_classes = 256
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1)
        data.train_mask = torch.ones(n).long()
    elif graphname == 'subgraph3':
        # path = "/gpfs/alpine/bif115/scratch/alokt/HipMCL/"
        # edge_index = torch.load(path + "/processed/subgraph3_graph.pt")
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
        data.y = torch.rand(n).uniform_(0, num_classes - 1)
        data.train_mask = torch.ones(n).long()

    if download:
        exit()

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
    # dist.init_process_group('gloo', init_method='env://')
    rank = dist.get_rank()
    size = dist.get_world_size()

    mp.set_start_method('spawn', force=True)
    # device = torch.device('cpu')
    devid = rank_to_devid(rank, acc_per_rank)

    device = torch.device('cuda:{}'.format(devid))
    print("device: " + str(device), flush=True)
    torch.cuda.set_device(device)
    curr_devid = torch.cuda.current_device()
    devcount = torch.cuda.device_count()
    print(f"devid: {curr_devid} {devcount}", flush=True)

    if graphname == "Amazon":
        # edge_index = edge_index.to(device)
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        data = data.to(device)
        # inputs = inputs.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif graphname == "subgraph5":
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif graphname == "subgraph3":
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    else:
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)

        edge_index = data.edge_index

    if normalization:
        adj_matrix, _ = add_remaining_self_loops(edge_index)
    else:
        adj_matrix = edge_index

    outputs = None
    print("Processes: " + str(size), flush=True)

    # init_process(rank, size, inputs, adj_matrix, data, dataset.num_features, mid_layer, dataset.num_classes, 
    init_process(rank, size, inputs, adj_matrix, data, num_features, mid_layer, num_classes,
                    device, outputs, acc_per_rank, run)

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
    parser.add_argument("--normalization", type=str)
    parser.add_argument("--activations", type=str)
    parser.add_argument("--accuracy", type=str)
    parser.add_argument("--download", type=bool)
    args = parser.parse_args()
    print(args)

    acc_per_rank = args.accperrank
    if acc_per_rank is None:
        acc_per_rank = 1

    epochs = args.epochs
    graphname = args.graphname
    timing = args.timing == "True"
    mid_layer = args.midlayer
    run_count = args.runcount
    normalization = args.normalization == "True"
    activations = args.activations == "True"
    accuracy = args.accuracy == "True"
    download = args.download

    if not download:
        if (epochs is None) or (graphname is None) or (timing is None) or (mid_layer is None) or (run_count is None):
            print(f"Error: missing argument {epochs} {graphname} {timing} {mid_layer}")
            exit()

    print(f"Arguments: epochs: {epochs} graph: {graphname} timing: {timing} mid: {mid_layer} norm: {normalization} act: {activations} acc: {accuracy}")
    
    print(main())
