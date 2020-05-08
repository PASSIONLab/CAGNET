import os
import os.path as osp
import argparse

import math

import torch
import torch_sparse
import torch.distributed as dist

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI, Reddit
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

from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_add

import socket
import time
import numpy as np

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

comp_time = 0.0
comm_time = 0.0
summa_sparse_bcast1 = 0.0
summa_sparse_bcast1_words = 0
summa_sparse_bcast2_words = 0
summa_sparse_bcast2 = 0.0
summa_sparse_bcast2_fwd = 0.0
summa_sparse_bcast2_bwd = 0.0
summa_bcast1 = 0.0
summa_bcast2 = 0.0
summa_sparse_comp = 0.0
summa_comp = 0.0
summa_loc_bcast = 0.0
fwd_time = 0.0
bwd_time = 0.0
transpose_time = 0.0
grad_weight_time = 0.0
loss_calc_time = 0.0
summa_sparse_time = 0.0
summa_time = 0.0
summa_loc_time = 0.0

epochs = 0
graphname = ""
mid_layer = 0
timing = False
normalization = False
no_occur_val = 42.1234

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
    if not timing:
        return 0.0
    dist.barrier(group)
    tstart = 0.0
    if rank == 0:
     tstart = time.time()
    return tstart

def stop_time(group, rank, tstart):
    if not timing:
        return 0.0
    dist.barrier(group)
    tstop = 0.0
    if rank == 0:
        tstop = time.time()
    return tstop - tstart

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

    # h = torch.exp(z - maxes)
    sm_sum = torch.sum(torch.exp(z - maxes), dim=1, keepdim=True)

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

    # width_per_proc = int(math.ceil(float(width) / proc_col))
    width_per_proc = width // proc_col
    leftover = width - width_per_proc * proc_col
    if z.size(1) != width_per_proc + leftover:
        z = torch.cat((z, torch.cuda.FloatTensor(z.size(0), leftover)), dim=1)

    z_recv = []
    for i in range(proc_col):
        z_recv.append(torch.cuda.FloatTensor(z.size()))

    dist.all_gather(z_recv, z, group=group)
    z_recv[rank_col] = z

    for i in range(len(z_recv)):
        if i < proc_col - 1 and leftover > 0:
            z_recv[i] = z_recv[i][:,:-leftover]

    z = torch.cat(z_recv, dim=1)

    if grad_output is not None:
        if grad_output.size(1) != width_per_proc + leftover:
            grad_output = torch.cat((grad_output, 
                                        torch.cuda.FloatTensor(grad_output.size(0), 
                                                        leftover)), 
                                        dim=1)

        grad_output_recv = []
        for i in range(proc_col):
            grad_output_recv.append(torch.cuda.FloatTensor(grad_output.size()))

        dist.all_gather(grad_output_recv, grad_output, group=group)
        grad_output_recv[rank_col] = grad_output

        for i in range(len(grad_output_recv)):
            if i < proc_col - 1 and leftover > 0:
                grad_output_recv[i] = grad_output_recv[i][:,:-leftover]

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
    else:
        dist.broadcast(mat_recvs[1], src=rank_t, group=transpose_group)
        dist.broadcast(mat_recvs[0], src=rank, group=transpose_group)

    return mat_recvs[1]

def summa(adj_matrix, inputs, rank, row, col, size, acc_per_rank, row_groups, col_groups, height, 
            middim, width):

    global comm_time
    global comp_time

    global summa_bcast1
    global summa_bcast2

    global summa_comp
    global summa_time

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

        dist.broadcast(acol.contiguous(), row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time += dur
        summa_bcast1 += dur

        if col_src_rank == rank:
            brow = inputs
        else:
            brow = brow_tens
            # brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=device)

        tstart = start_time(col_groups[col], rank)

        dist.broadcast(brow.contiguous(), col_src_rank, col_groups[col])

        dur = stop_time(col_groups[col], rank, tstart)
        comm_time += dur
        summa_bcast2 += dur

        tstart = start_time(row_groups[0], rank)

        z_loc += torch.mm(acol.float(), brow)

        dur = stop_time(row_groups[0], rank, tstart)
        comp_time += dur
        summa_comp += dur

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
        # dist.broadcast(acol_values_len, row_src_rank, row_groups[row])

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

        dist.broadcast(acol, row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time += dur
        summa_sparse_bcast1 += dur
        if rank == 0:
            summa_sparse_bcast1_words += 3 * acol_values_len

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

        dist.broadcast(brow, col_src_rank, col_groups[col])

        dur = stop_time(row_groups[0], rank, tstart)
        comm_time += dur
        summa_sparse_bcast2 += dur
        if rank == 0:
            summa_sparse_bcast2_words += brow.size(0) * brow.size(1)

        tstart = start_time(row_groups[0], rank)

        spmm_gpu(acol_indices[0].int(), acol_indices[1].int(), acol_values, 
                        height_per_proc, middim_per_proc, brow, z_loc)

        dur = stop_time(row_groups[0], rank, tstart)
        comp_time += dur
        summa_sparse_comp += dur

    # summa_sparse_time += stop_time(row_groups[0], rank, tstart_summa_sparse_time)
    return z_loc

def summa_loc(mata, matb, rank, row, col, size, acc_per_rank, row_groups, col_groups, 
                    height, middim, width):

    global comm_time
    global comp_time

    global summa_loc_bcast
    global summa_loc_time

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

        dist.broadcast(acol.contiguous(), row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time += dur
        summa_loc_bcast += dur

        # if col_src_rank == rank:
        #     brow = matb.clone()
        # else:
        #     brow = torch.FloatTensor(middim_per_proc, width_per_proc)

        # dist.broadcast(brow, col_src_rank, col_groups[col])

        brow = matb[col_src_rank]

        tstart = start_time(row_groups[0], rank)

        z_loc += torch.mm(acol, brow)

        dur = stop_time(row_groups[0], rank, tstart)
        comp_time += dur

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

        tmp_summa_sparse_bcast2 = summa_sparse_bcast2

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

        summa_sparse_bcast2_fwd += summa_sparse_bcast2 - tmp_summa_sparse_bcast2

        # Worry about activation later
        if func is F.log_softmax:
            h = dist_log_softmax(z, rank, size, acc_per_rank, row_groups[rank_row])
            # h = dist_log_softmax2(z, rank, size, weight.size(1), acc_per_rank, 
            #                        row_groups[rank_row], None)
        elif func is F.relu:
            h = func(z)
        else:
            h = z

        # dur = stop_time(row_groups[0], rank, tstart)
        # fwd_time += dur

        # return z
        return h

    @staticmethod
    def backward(ctx, grad_output):
        global summa_sparse_bcast2
        global summa_sparse_bcast2_bwd
        global bwd_time
        global transpose_time
        global grad_weight_time

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
            
        # Worry about activation later
        with torch.set_grad_enabled(True):
            if func is F.log_softmax:
                # func_eval = dist_log_softmax(z, rank, size, acc_per_rank, row_groups[rank_row])
                func_eval, z_gathered, go_gathered = dist_log_softmax2(z, rank, size, 
                                                                weight.size(1), 
                                                                acc_per_rank, 
                                                                row_groups[rank_row], grad_output)
                width = z_gathered.size(1)

                sigmap = torch.autograd.grad(outputs=func_eval, inputs=z_gathered,
                                                grad_outputs=go_gathered)[0]

                chunk_sizes = []
                width_per_col = width // proc_col
                for i in range(proc_col):
                    if i == proc_col - 1:
                        chunk_sizes.append(width - width_per_col * (proc_col - 1))
                    else:
                        chunk_sizes.append(width_per_col)

                grad_output = sigmap.split(chunk_sizes, dim=1)[rank_col]
                del z_gathered
                del go_gathered
            elif func is F.relu:
                func_eval = func(z)
                sigmap = torch.autograd.grad(outputs=func_eval, inputs=z,
                                                grad_outputs=grad_output)[0]
                grad_output = sigmap
            else:
                func_eval = z
                sigmap = torch.autograd.grad(outputs=func_eval, inputs=z,
                                                grad_outputs=grad_output)[0]
                grad_output = sigmap


            # print(f"rank: {rank} sigmap: {sigmap}", flush=True)
            del func_eval

        tmp_summa_sparse_bcast2 = summa_sparse_bcast2

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

        tstart_transpose = start_time(row_groups[0], rank)
        inputs_t = transpose(inputs, rank_row, rank_col, node_count, weight.size(0), size,
                                acc_per_rank, transpose_group)
        transpose_time += stop_time(row_groups[0], rank, tstart_transpose)

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

        summa_sparse_bcast2_bwd += summa_sparse_bcast2 - tmp_summa_sparse_bcast2

        # dur = stop_time(row_groups[0], rank, tstart)
        # bwd_time += dur

        # grad_weight_time += stop_time(row_groups[0], rank, tstart_grad_weight)

        return grad_input, grad_weight_fin, None, None, None, None, None, None, None, None, None, None, None

def train(inputs, weight1, weight2, node_count, adj_matrix, am_partitions, optimizer, data, rank, 
                size, acc_per_rank, group, row_groups, col_groups, transpose_group):

    global loss_calc_time

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
        tstart_loss_calc = start_time(row_groups[0], rank)

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

        dist.reduce(loss_calc, dst=rank_row_src, op=dist.reduce_op.SUM, group=row_groups[rank_row])
        dist.broadcast(loss_calc, src=rank_row_src, group=row_groups[rank_row]) 

        vertex_train_count = (data.train_mask.size(0) - (data.train_mask == 0).sum(dim=0))
        loss_calc = -loss_calc / vertex_train_count

        loss_calc_time += stop_time(row_groups[0], rank, tstart_loss_calc)
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
    datay_rank = torch.split(data.y, vertex_count)[rank]
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        mask_rank = torch.split(mask, vertex_count)[rank]
        count = mask_rank.nonzero().size(0)
        if count > 0:
            pred = logits[mask_rank].max(1)[1]
            acc = pred.eq(datay_rank[mask_rank]).sum().item() / mask_rank.sum().item()
            # pred = logits[mask].max(1)[1]
            # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        else:
            acc = -1
        accs.append(acc)
    return accs


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
        return

    # Scale each edge (u, v) by 1 / (sqrt(u) * sqrt(v))
    indices = adj_part._indices()
    values = adj_part._values()

    deg_map = dict()
    for i in range(adj_part._nnz()):
        u = indices[0][i] + row_vtx
        v = indices[1][i] + col_vtx

        if u.item() in deg_map:
            degu = deg_map[u.item()]
        else:
            degu = (adj_matrix[0] == u).sum().item()
            deg_map[u.item()] = degu

        if v.item() in deg_map:
            degv = deg_map[v.item()]
        else:
            degv = (adj_matrix[0] == v).sum().item()
            deg_map[v.item()] = degv

        values[i] = values[i] / (math.sqrt(degu) * math.sqrt(degv))
    
    # deg = torch.histc(adj_matrix[0].float(), bins=node_count)
    # deg = deg.pow(-0.5)

    # row_len = adj_part.size(0)
    # col_len = adj_part.size(1)

    # dleft = torch.sparse_coo_tensor([np.arange(row_vtx, row_vtx + row_len).tolist(),
    #                                  np.arange(row_vtx, row_vtx + row_len).tolist()],
    #                                  deg[row_vtx:(row_vtx + row_len)],
    #                                  size=(row_len, row_len),
    #                                  requires_grad=False)

    # dright = torch.sparse_coo_tensor([np.arange(col_vtx, col_vtx + col_len).tolist(),
    #                                  np.arange(col_vtx, col_vtx + col_len).tolist()],
    #                                  deg[row_vtx:(col_vtx + col_len)],
    #                                  size=(col_len, col_len),
    #                                  requires_grad=False)

    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    # return adj_part

def proc_row_size(size):
    return math.floor(math.sqrt(size))

def proc_col_size(size):
    return math.floor(math.sqrt(size))

def twod_partition(rank, size, inputs, adj_matrix, data, features, classes, device):
    node_count = inputs.size(0)
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

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

                scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                    vtx_indices[rank_col])
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
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

    best_val_acc = test_acc = 0
    outputs = None

    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size, group)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    rank_t  = rank_col * proc_row + rank_row

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

    # tstart = start_time(group, rank)
    if rank == 0:
        tstart = time.time()

    for epoch in range(epochs):
        if rank == 0:
            tstart_epoch = time.time()
        outputs = train(inputs_loc, weight1, weight2, inputs.size(0), adj_matrix_loc, None, 
                                optimizer, data, rank, size, acc_per_rank, group, row_groups, 
                                col_groups, transpose_group)
        if rank == 0 and timing:
            tstop_epoch = time.time()
            tstop = time.time()
            print(f"Epoch time: {epoch} {tstop_epoch - tstart_epoch}", flush=True)
            print("Total Time: " + str(tstop - tstart))
            print("comm_time: " + str(comm_time))
            print("comp_time: " + str(comp_time))
            print(f"summa_sparse_comp: {summa_sparse_comp}")
            print(f"summa_sparse_bcast1: {summa_sparse_bcast1}")
            print(f"summa_sparse_bcast1_words: {summa_sparse_bcast1_words}")
            print(f"summa_sparse_bcast2: {summa_sparse_bcast2}")
            print(f"summa_sparse_bcast2_fwd: {summa_sparse_bcast2_fwd}")
            print(f"summa_sparse_bcast2_bwd: {summa_sparse_bcast2_bwd}")
            print(f"summa_sparse_bcast2_words: {summa_sparse_bcast2_words}")
            print(f"summa_comp: {summa_comp}")
            print(f"summa_bcast1: {summa_bcast1}")
            print(f"summa_bcast2: {summa_bcast2}")
            print(f"summa_loc_bcast: {summa_loc_bcast}")
            print(f"fwd_time: {fwd_time}")
            print(f"bwd_time: {bwd_time}")
            print(f"transpose_time: {transpose_time}")
            print(f"grad_weight_time: {grad_weight_time}")
            print(f"loss_calc_time: {loss_calc_time}")
            print(f"summa_sparse_time: {summa_sparse_time}")
            print(f"summa_time: {summa_time}")
            print("Epoch: {:03d}".format(epoch), flush=True)

    # dur = stop_time(group, rank, tstart)
    if rank == 0:
        tstop = time.time()
        print("Time: " + str(tstop - tstart))

    if rank == 0 and timing:
        print("comm_time: " + str(comm_time))
        print("comp_time: " + str(comp_time))
        print(f"summa_sparse_comp: {summa_sparse_comp}")
        print(f"summa_sparse_bcast1: {summa_sparse_bcast1}")
        print(f"summa_sparse_bcast1_words: {summa_sparse_bcast1_words}")
        print(f"summa_sparse_bcast2: {summa_sparse_bcast2}")
        print(f"summa_sparse_bcast2_fwd: {summa_sparse_bcast2_fwd}")
        print(f"summa_sparse_bcast2_bwd: {summa_sparse_bcast2_bwd}")
        print(f"summa_sparse_bcast2_words: {summa_sparse_bcast2_words}")
        print(f"summa_comp: {summa_comp}")
        print(f"summa_bcast1: {summa_bcast1}")
        print(f"summa_bcast2: {summa_bcast2}")
        print(f"summa_loc_bcast: {summa_loc_bcast}")
        print(f"fwd_time: {fwd_time}")
        print(f"bwd_time: {bwd_time}")
        print(f"transpose_time: {transpose_time}")
        print(f"grad_weight_time: {grad_weight_time}")
        print(f"loss_calc_time: {loss_calc_time}")
        print(f"summa_sparse_time: {summa_sparse_time}")
        print(f"summa_time: {summa_time}")
        print(f"summa_loc_time: {summa_loc_time}")
    
    # All-gather outputs to test accuracy
    # output_parts = []
    # for i in range(size):
    #     output_parts.append(torch.cuda.FloatTensor(am_partitions[0].size(1), classes).fill_(0))

    # dist.all_gather(output_parts, outputs)
    # outputs = torch.cat(output_parts, dim=0)

    # train_acc, val_acc, tmp_test_acc = test(outputs, data, am_partitions[0].size(1), rank)
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

    # print(log.format(200, train_acc, best_val_acc, test_acc))
    print("rank: " + str(rank) + " " +  str(outputs))
    return outputs

def init_process(rank, size, inputs, adj_matrix, data, features, mid_layer, classes, device, 
                    outputs, acc_per_rank, fn):

    run_outputs = fn(rank, size, inputs, adj_matrix, data, features, mid_layer, classes, device, 
                            acc_per_rank)
    if outputs is not None:
        outputs[rank] = run_outputs.detach()

def main(P, correctness_check, acc_per_rank):
    # graphname = 'Reddit'
    global graphname
    global mid_layer

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)

    # mid_layer = 16
    if graphname == 'Cora':
        dataset = Planetoid(path, graphname, T.NormalizeFeatures())
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif graphname == 'Reddit':
        dataset = Reddit(path, T.NormalizeFeatures())
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes + 9
    elif graphname == 'Amazon':
        edge_index = torch.load(path + "/processed/amazon_graph.pt")
        edge_index = edge_index.t_()
        n = 9430086
        num_features = 300
        num_classes = 24
        # mid_layer = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1)
        data.train_mask = torch.ones(n).long()
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
        path = "/gpfs/alpine/bif115/scratch/alokt/HipMCL/"
        edge_index = torch.load(path + "/processed/subgraph3_graph.pt")
        n = 8745542
        num_features = 128
        # mid_layer = 512
        # mid_layer = 64
        num_classes = 256
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1)
        data.train_mask = torch.ones(n).long()

    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
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
        print("edge count: " + str(len(edge_index[0])))

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
    parser.add_argument('--processes', metavar='P', type=int,
                        help='Number of processes')
    parser.add_argument('--correctness', metavar='C', type=str,
                        help='Run correctness check')
    parser.add_argument("--accperrank", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--graphname", type=str)
    parser.add_argument("--timing", type=str)
    parser.add_argument("--midlayer", type=int)
    args = parser.parse_args()
    print(args)
    P = args.processes
    correctness_check = args.correctness
    if P is None:
        P = 1

    acc_per_rank = args.accperrank
    if acc_per_rank is None:
        acc_per_rank = 1

    if correctness_check is None or correctness_check == "nocheck":
        correctness_check = False
    else:
        correctness_check = True

    epochs = args.epochs
    graphname = args.graphname
    timing = args.timing == "True"
    mid_layer = args.midlayer

    if (epochs is None) or (graphname is None) or (timing is None) or (mid_layer is None):
        print(f"Error: missing argument {epochs} {graphname} {timing} {mid_layer}")
        exit()

    print(f"Arguments: epochs: {epochs} graph: {graphname} timing: {timing} mid: {mid_layer}")
    
    print("Correctness: " + str(correctness_check))
    print(main(P, correctness_check, acc_per_rank))
