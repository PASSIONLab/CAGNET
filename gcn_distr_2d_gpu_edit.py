import os
import os.path as osp
import argparse

import math

import torch
import torch_sparse
import torch.distributed as dist

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

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu

comp_time = 0.0
comm_time = 0.0

normalization = True
no_occur_val = 42.1234

def normalize(adj_matrix):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    d = torch.sum(adj_matrix, dim=1)
    d = torch.rsqrt(d)
    d = torch.diag(d)
    return torch.mm(d, torch.mm(adj_matrix, d))

def start_time(group, rank):
    # dist.barrier(group)
    # tstart = 0.0
    # if rank == 0:
    #     tstart = time.time()
    # return tstart
    return 0.0

def stop_time(group, rank, tstart):
    # dist.barrier(group)
    # tstop = 0.0
    # if rank == 0:
    #     tstop = time.time()
    # return tstop - tstart
    return 0.0

def intersect(t1, t2):
    t1_t = t1.t().squeeze()
    t2_t = t2.t().squeeze()
    indices = torch.zeros_like(t1_t, dtype = torch.uint8, device = 'cuda')
    for elem in t2_t:
        indices = indices | (t1_t == elem).byte()
    intersection = t1_t[indices]
    return intersection.t()

def transpose(mat, row, col, height, width, size):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    rank = row * proc_col + col
    rank_t  = col * proc_row + row

    if rank == rank_t:
        return mat.t()

    height_recv = math.ceil(float(width) / proc_row)
    width_recv  = math.ceil(float(height) / proc_col)

    if row == proc_row - 1:
        height_recv -= proc_row * height_recv - width

    if col == proc_col - 1:
        width_recv -= proc_col * width_recv - height

    mat_recv = torch.cuda.FloatTensor(height_recv, width_recv)

    # if rank < rank_t:
    #     dist.send(tensor=mat.t().contiguous(), dst=rank_t)
    #     dist.recv(tensor=mat_recv, src=rank_t)
    # else:
    #     dist.recv(tensor=mat_recv, src=rank_t)
    #     dist.send(tensor=mat.t().contiguous(), dst=rank_t)

    transpose_group = dist.new_group([rank, rank_t])

    mat_recvs = [mat.t().contiguous(), mat_recv]

    if rank < rank_t:
        dist.broadcast(mat_recvs[0], src=rank, group=transpose_group)
        dist.broadcast(mat_recvs[1], src=rank_t, group=transpose_group)
    else:
        dist.broadcast(mat_recvs[1], src=rank_t, group=transpose_group)
        dist.broadcast(mat_recvs[0], src=rank, group=transpose_group)


    return mat_recvs[1]

def summa(adj_matrix, inputs, rank, row, col, size, row_groups, col_groups, height, middim, 
            width, transpose):

    global comm_time
    global comp_time

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    height_per_proc = math.ceil(float(height) / proc_row)
    width_per_proc  = math.ceil(float(width) / proc_col)
    # TODO: Not sure how to handle this w/o square grid
    middim_per_proc = math.ceil(float(middim) / proc_row)

    if row == proc_row - 1:
        height_per_proc -= proc_row * height_per_proc - height

    if col == proc_col - 1:
        width_per_proc -= proc_col * width_per_proc - width

    acol = torch.cuda.FloatTensor(height_per_proc, middim_per_proc)

    brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc).fill_(0)

    for k in range(proc_col):

        if transpose:
            row_src_rank = k * proc_row + row
            col_src_rank = k * proc_col + col
        else:
            row_src_rank = k + proc_col * row
            col_src_rank = k * proc_col + col

        if k == proc_col - 1:
            middim_per_proc -= proc_col * middim_per_proc - middim

        if row_src_rank == rank:
            acol = adj_matrix
        else:
            acol = torch.cuda.FloatTensor(height_per_proc, middim_per_proc)
        
        tstart = start_time(row_groups[row], rank)

        dist.broadcast(acol.contiguous(), row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time += dur

        if col_src_rank == rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc)

        tstart = start_time(col_groups[col], rank)

        dist.broadcast(brow.contiguous(), col_src_rank, col_groups[col])

        dur = stop_time(col_groups[col], rank, tstart)
        comm_time += dur

        tstart = start_time(row_groups[0], rank)

        z_loc += torch.mm(acol.float(), brow)

        dur = stop_time(row_groups[0], rank, tstart)
        comp_time += dur

    return z_loc

def summa_sparse(adj_matrix, inputs, rank, row, col, size, row_groups, col_groups, height, middim, 
                    width, transpose):

    global comm_time
    global comp_time

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    height_per_proc = math.ceil(float(height) / proc_row)
    width_per_proc  = math.ceil(float(width) / proc_col)

    # TODO: Not sure how to handle this w/o square grid
    middim_per_proc = math.ceil(float(middim) / proc_col)

    if row == proc_row - 1:
        height_per_proc -= proc_row * height_per_proc - height

    if col == proc_col - 1:
        width_per_proc -= proc_col * width_per_proc - width

    acol = torch.cuda.sparse.FloatTensor(height_per_proc, middim_per_proc)

    brow = torch.FloatTensor(middim_per_proc, width_per_proc)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc).fill_(0)

    for k in range(proc_col):

        if transpose:
            row_src_rank = k * proc_row + row
            col_src_rank = k * proc_col + col
        else:
            row_src_rank = k + proc_col * row
            col_src_rank = k * proc_col + col

        if k == proc_col - 1:
            middim_per_proc -= proc_col * middim_per_proc - middim

        if row_src_rank == rank:
            # acol = adj_matrix.clone()
            acol_indices_len = torch.cuda.LongTensor([adj_matrix.indices().contiguous()[0].size(0)])
            acol_values_len = torch.cuda.LongTensor([adj_matrix.values().contiguous().size(0)])
        else:
            # acol = torch.sparse.FloatTensor(height_per_proc, middim_per_proc)
            acol_indices_len = torch.cuda.LongTensor([0])
            acol_values_len = torch.cuda.LongTensor([0])

        dist.broadcast(acol_indices_len, row_src_rank, row_groups[row])
        dist.broadcast(acol_values_len, row_src_rank, row_groups[row])

        acol_indices_len = acol_indices_len.item()
        acol_values_len = acol_values_len.item()

        if row_src_rank == rank:
            acol_indices = adj_matrix.indices().contiguous().long()
            acol_values = adj_matrix.values().contiguous().float()
        else:
            acol_indices = torch.cuda.LongTensor(2, acol_indices_len).fill_(0)
            acol_values = torch.cuda.FloatTensor(acol_values_len).fill_(0)
        
        tstart = start_time(row_groups[row], rank)

        dist.broadcast(acol_indices[0], row_src_rank, row_groups[row])
        dist.broadcast(acol_indices[1], row_src_rank, row_groups[row])
        dist.broadcast(acol_values, row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time += dur

        if row_src_rank == rank:
            acol = adj_matrix
        else:
            acol = sparse_coo_tensor_gpu(acol_indices, acol_values, 
                                            torch.Size([height_per_proc, middim_per_proc]))

        if col_src_rank == rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc)

        tstart = start_time(row_groups[0], rank)

        dist.broadcast(brow.contiguous(), col_src_rank, col_groups[col])

        dur = stop_time(row_groups[0], rank, tstart)
        comm_time += dur
        
        tstart = start_time(row_groups[0], rank)

        z_tmp = torch.sparse.mm(acol, brow)

        dur = stop_time(row_groups[0], rank, tstart)
        comp_time += dur

        z_loc += z_tmp

    return z_loc

def summa_loc(mata, matb, rank, row, col, size, row_groups, col_groups, 
                    height, middim, width, transpose):

    global comm_time
    global comp_time

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    height_per_proc = math.ceil(float(height) / proc_row)
    width_per_proc  = math.ceil(float(width) / proc_col)
    # TODO: Not sure how to handle this w/o square grid
    middim_per_proc = math.ceil(float(middim) / proc_row)

    if row == proc_row - 1:
        height_per_proc -= proc_row * height_per_proc - height

    # if col == proc_col - 1:
    #     width_per_proc -= proc_col * width_per_proc - width
    width_per_proc = matb[rank].size(1)

    acol = torch.cuda.FloatTensor(height_per_proc, middim_per_proc)

    brow = torch.FloatTensor(middim_per_proc, width_per_proc)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc).fill_(0)

    for k in range(proc_col):

        if transpose:
            row_src_rank = k * proc_row + row
            col_src_rank = k * proc_col + col
        else:
            row_src_rank = k + proc_col * row
            col_src_rank = k * proc_col + col

        if k == proc_col - 1:
            middim_per_proc -= proc_col * middim_per_proc - middim

        if row_src_rank == rank:
            acol = mata
        else:
            acol = torch.cuda.FloatTensor(height_per_proc, matb[col_src_rank].size(0))
        
        tstart = start_time(row_groups[row], rank)

        dist.broadcast(acol.contiguous(), row_src_rank, row_groups[row])

        dur = stop_time(row_groups[row], rank, tstart)
        comm_time += dur

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

    return z_loc

def get_proc_groups(rank, size):
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    
    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
        
    row_groups = []
    col_groups = []

    for i in range(proc_row):
        row_groups.append(dist.new_group(list(range(i * proc_col, i * proc_col + proc_col))))

    for i in range(proc_col):
        col_groups.append(dist.new_group(list(range(i, size, proc_row))))

    return row_groups, col_groups

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, node_count, adj_matrix, am_partitions, rank, size, group, 
                        row_groups, col_groups, func):
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        proc_row = proc_row_size(size)
        proc_col = proc_col_size(size)
        
        rank_row = int(rank / proc_col)
        rank_col = rank % proc_col

        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.node_count = node_count
        ctx.rank = rank
        ctx.size = size
        ctx.group = group
        ctx.row_groups = row_groups
        ctx.col_groups = col_groups

        ctx.func = func

        adj_matrix_t = adj_matrix # Only true for undirected graphs

        # TODO: will need to change height argument when n % sqrt(P) != 0 and non-square grid
        z = summa_sparse(adj_matrix_t, inputs, rank, rank_row, rank_col, size, row_groups, 
                            col_groups, node_count, node_count, weight.size(0), transpose=False)

        weight_rows = torch.split(weight, math.ceil(float(weight.size(0)) / proc_row), dim=0)
        weight_parts = []
        for i in weight_rows:
            weight_cols = torch.split(i, math.ceil(float(weight.size(1)) / proc_col), dim=1)
            weight_parts.extend(weight_cols)

        # z = torch.mm(z, weight)
        z = summa_loc(z, weight_parts, rank, rank_row, rank_col, size, row_groups, col_groups, 
                        node_count, weight.size(0), weight.size(1), transpose=False)

        z.requires_grad = True
        ctx.z = z

        # Worry about activation later
        # if func is F.log_softmax:
        #     h = func(z, dim=1)
        # elif func is F.relu:
        #     h = func(z)
        # else:
        #     h = z

        # return h
        return z

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors
        rank = ctx.rank
        size = ctx.size
        group = ctx.group
        row_groups = ctx.row_groups
        col_groups = ctx.col_groups
        node_count = ctx.node_count

        func = ctx.func
        z = ctx.z

        # Worry about activation later
        # with torch.set_grad_enabled(True):
        #     if func is F.log_softmax:
        #         func_eval = func(z, dim=1)
        #     elif func is F.relu:
        #         func_eval = func(z)
        #     else:
        #         func_eval = z

        #     sigmap = torch.autograd.grad(outputs=func_eval, inputs=z,grad_outputs=grad_output)[0]
        #     grad_output = sigmap

        proc_row = proc_row_size(size)
        proc_col = proc_col_size(size)

        rank_row = int(rank / proc_col)
        rank_col = rank % proc_col
            
        # First backprop equation
        # TODO: will need to change height argument when n % sqrt(P) != 0 and non-square grid
        ag = summa_sparse(adj_matrix, grad_output, rank, rank_row, rank_col, size, row_groups, 
                            col_groups, node_count, node_count, weight.t().size(0), 
                            transpose=False)

        weight_rows = torch.split(weight.t(), math.ceil(float(weight.t().size(0)) / proc_row), 
                                        dim=0)
        weight_parts = []
        for i in weight_rows:
            weight_cols = torch.split(i, math.ceil(float(weight.t().size(1)) / proc_col), dim=1)
            weight_parts.extend(weight_cols)

        # grad_input = torch.mm(ag, weight.t())
        grad_input = summa_loc(ag, weight_parts, rank, rank_row, rank_col, size, row_groups, 
                                    col_groups, node_count, weight.t().size(0), weight.t().size(1),
                                    transpose=False)

        # Second backprop equation (reuses the A * G^l computation)
        # col_groups twice because of transpose
        # TODO: will need to change height argument when n % sqrt(P) != 0 and non-square grid

        inputs_t = transpose(inputs, rank_row, rank_col, node_count, weight.size(0), size)

        grad_weight = summa(inputs_t, ag, rank, rank_row, rank_col, size, row_groups, col_groups, 
                                weight.size(0), node_count, weight.size(1), transpose=False)

        # Collect grad_weight's across processes
        grad_weight_recv = []
        for i in range(size):
            grad_weight_recv.append(torch.cuda.FloatTensor(math.ceil(float(weight.size(0)) / proc_row),
                                                math.ceil(float(weight.size(1)) / proc_col)))

        pad_row = math.ceil(float(weight.size(0)) / proc_row) - grad_weight.size(0)
        pad_col = math.ceil(float(weight.size(1)) / proc_col) - grad_weight.size(1)

        # TODO: make this part less hacky
        grad_weight = torch.cat((grad_weight, 
                                torch.cuda.FloatTensor(pad_row, grad_weight.size(1)).fill_(no_occur_val)), 
                                dim=0) 
        grad_weight = torch.cat((grad_weight, 
                                torch.cuda.FloatTensor(grad_weight.size(0), pad_col).fill_(no_occur_val)), 
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
        
        grad_weight_fin = torch.cuda.FloatTensor()
        for i in range(proc_row):
            grad_weight_row = torch.cuda.FloatTensor()
            for j in range(proc_col):
                rank_wt = i * proc_row + j
                grad_weight_row = torch.cat((grad_weight_row, grad_weight_recv[rank_wt]), dim=1)
            grad_weight_fin = torch.cat((grad_weight_fin, grad_weight_row), dim=0)

        return grad_input, grad_weight_fin, None, None, None, None, None, None, None, None, None

def train(inputs, weight1, weight2, node_count, adj_matrix, am_partitions, optimizer, data, rank, 
                size, group, row_groups, col_groups):

    outputs = GCNFunc.apply(inputs, weight1, node_count, adj_matrix, am_partitions, rank, size, 
                                    group, row_groups, col_groups, F.relu)

    outputs = GCNFunc.apply(outputs, weight2, node_count, adj_matrix, am_partitions, rank, size, 
                                    group, row_groups, col_groups, F.log_softmax)

    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask, outputs.size(0), dim=0)[rank_row]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[rank_row]

    total_classes = weight2.size(1)
    class_per_rank = math.ceil(float(total_classes) / proc_col)

    min_class = rank * class_per_rank
    max_class = min((rank + 1) * class_per_rank, total_classes)

    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
    # if datay_rank.size(0) > 0:
        # datay_ids = datay_rank[rank_train_mask].long().view(-1, 1)
        datay_ids = datay_rank[rank_train_mask].long()

        filtered_indices = torch.mul(datay_ids >= min_class, datay_ids < max_class).float()
        indices = torch.nonzero(filtered_indices * torch.cuda.FloatTensor(datay_ids.size()).fill_(1)).squeeze()


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

        loss_calc.backward()
        # print("loss_calc: " + str(loss_calc), flush=True)
        # loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        # loss.backward()
        # print("loss: " + str(loss), flush=True)
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size()).fill_(0)).sum()
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
def scale_elements(adj_part, node_count, rank, size, row, col, row_vtx, col_vtx):
    if not normalization:
        return
    
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)
    row_groups, col_groups = get_proc_groups(rank, size)

    # Compute degrees across process row
    degs = torch.sparse.sum(adj_part, dim=1).to_dense().to(torch.device("cuda"))
    dist.all_reduce(degs, op=dist.reduce_op.SUM, group=row_groups[row]) 

    # Gather degrees across process col
    # TODO: modify when node_count % proc_row != 0
    recv_row_size = int(math.ceil(float(node_count) / proc_row))

    degs_list = []
    for i in range(proc_row):
        degs_list.append(torch.zeros(recv_row_size).to(torch.device("cuda")))

    dist.all_gather(degs_list, degs, group=col_groups[col])
    for i in range(proc_row):
        degs_list[i] = degs_list[i].to(torch.device("cpu"))

    degs = torch.cat(degs_list)

    height_per_proc = math.ceil(float(node_count) / proc_row)
    width_per_proc  = math.ceil(float(node_count) / proc_col)

    if row == proc_row - 1:
        height_per_proc -= proc_row * height_per_proc - node_count

    if col == proc_col - 1:
        width_per_proc -= proc_col * width_per_proc - node_count

    min_vtx = max(row_vtx, col_vtx)
    max_vtx = min(row_vtx + height_per_proc, col_vtx + width_per_proc)

    degs_values = degs[min_vtx:max_vtx]
    degs_values = degs_values.pow(-0.5)

    degs_indices_src = torch.Tensor(np.arange(min_vtx - row_vtx, max_vtx - row_vtx)).unsqueeze(0)
    degs_indices_dst = torch.Tensor(np.arange(min_vtx - col_vtx, max_vtx - col_vtx)).unsqueeze(0)

    degs_indices = torch.cat((degs_indices_src, degs_indices_dst), dim=0)

    degs = torch.sparse_coo_tensor(degs_indices, degs_values, size=adj_part.size(), 
                                        requires_grad=False)
    
    degs = degs.coalesce()
    adj_part = adj_part.coalesce()
    indices_inner, values_inner = torch_sparse.spspmm(adj_part.indices(), adj_part.values(),
                                                          degs.indices(), degs.values(),
                                                          adj_part.size(0), adj_part.size(1),
                                                          degs.size(1))

    scaled_indices, scaled_values = torch_sparse.spspmm(degs.indices(), degs.values(),
                                                            indices_inner, values_inner,
                                                            degs.size(0), degs.size(1),
                                                            node_count)

    scaled_adj = torch.sparse_coo_tensor(scaled_indices, scaled_values, size=adj_part.size(),
                                            requires_grad=False)

    return scaled_adj

def proc_row_size(size):
    return math.floor(math.sqrt(size))

def proc_col_size(size):
    return math.floor(math.sqrt(size))

def twod_partition(rank, size, inputs, adj_matrix, data, features, classes, device):
    node_count = inputs.size(0)
    proc_row = proc_row_size(size)
    proc_col = proc_col_size(size)

    n_per_proc = math.ceil(float(node_count) / proc_row)

    rank_row = int(rank / proc_col)
    rank_col = rank % proc_col
    
    am_partitions = None
    am_pbyp = None

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[rank_col + 1] - vtx_indices[rank_col]
        am_pbyp, _ = split_coo(am_partitions[rank_col], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            if i == proc_row - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(am_pbyp[i], node_count, rank, size, rank_row, rank_col,
                                                vtx_indices[i], vtx_indices[rank_col])
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(am_pbyp[i], node_count, rank, size, rank_row, rank_col,
                                                vtx_indices[i], vtx_indices[rank_col])

        input_rowparts = torch.split(inputs, math.ceil(float(inputs.size(0)) / proc_row), dim=0)
        input_partitions = []
        for i in input_rowparts:
            input_partitions.append(torch.split(i, math.ceil(float(inputs.size(1)) / proc_col), 
                                        dim=1))

        adj_matrix_loc = am_pbyp[rank_row]
        inputs_loc = input_partitions[rank_row][rank_col]

    print(adj_matrix_loc.size(), flush=True)
    print(inputs_loc.size(), flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def run(rank, size, inputs, adj_matrix, data, features, classes, device):
    global comm_time
    global comp_time

    best_val_acc = test_acc = 0
    outputs = None
    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size)

    # adj_matrix_loc = torch.rand(node_count, n_per_proc)
    # inputs_loc = torch.rand(n_per_proc, inputs.size(1))

    torch.manual_seed(0)
    weight1_nonleaf = torch.rand(features, 16, requires_grad=True)
    weight1_nonleaf = weight1_nonleaf.to(device)
    weight1_nonleaf.retain_grad()

    weight2_nonleaf = torch.rand(16, classes, requires_grad=True)
    weight2_nonleaf = weight2_nonleaf.to(device)
    weight2_nonleaf.retain_grad()

    weight1 = Parameter(weight1_nonleaf)
    weight2 = Parameter(weight2_nonleaf)

    optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)

    inputs_loc, adj_matrix_loc, am_pbyp = twod_partition(rank, size, inputs, adj_matrix, data, 
                                                                features, classes, device)

    adj_matrix_loc = adj_matrix_loc.coalesce()


    inputs_loc = inputs_loc.to(device)
    adj_matrix_loc = adj_matrix_loc.to(device)

    for i in range(len(am_pbyp)):
        am_pbyp[i] = am_pbyp[i].to(device)
    # for epoch in range(1, 101):

    # tstart = start_time(group, rank)
    if rank == 0:
        tstart = time.time()

    for epoch in range(2):
        outputs = train(inputs_loc, weight1, weight2, inputs.size(0), adj_matrix_loc, am_pbyp, 
                                optimizer, data, rank, size, group, row_groups, col_groups)
    print("Epoch: {:03d}".format(epoch), flush=True)

    # dur = stop_time(group, rank, tstart)
    if rank == 0:
        tstop = time.time()
        print("Time: " + str(tstop - tstart))

    if rank == 0:
        print("comm_time: " + str(comm_time))
        print("comp_time: " + str(comp_time))
    
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

def init_process(rank, size, inputs, adj_matrix, data, features, classes, device, outputs, fn):
    run_outputs = fn(rank, size, inputs, adj_matrix, data, features, classes, device)
    if outputs is not None:
        outputs[rank] = run_outputs.detach()

def main(P, correctness_check):
    print(socket.gethostname())
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    # dataset = PPI(path, 'train', T.NormalizeFeatures())
    # dataset = Reddit(path, T.NormalizeFeatures())
    data = dataset[0]

    seed = 0

    mp.set_start_method('spawn', force=True)
    # device = torch.device('cpu')
    device = torch.device('cuda')
    print("device: " + str(device), flush=True)

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
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    dist.init_process_group(backend='nccl')
    # dist.init_process_group('gloo', init_method='env://')
    rank = dist.get_rank()
    size = dist.get_world_size()
    print("Processes: " + str(size), flush=True)

    init_process(rank, size, inputs, adj_matrix, data, dataset.num_features, dataset.num_classes, 
                    device, outputs, run)

    if outputs is not None:
        return outputs[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--processes', metavar='P', type=int,
                        help='Number of processes')
    parser.add_argument('--correctness', metavar='C', type=str,
                        help='Run correctness check')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    print(args)
    P = args.processes
    correctness_check = args.correctness
    if P is None:
        P = 1

    if correctness_check is None or correctness_check == "nocheck":
        correctness_check = False
    else:
        correctness_check = True
    
    print("Correctness: " + str(correctness_check))
    print(main(P, correctness_check))
