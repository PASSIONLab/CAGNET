import math
import torch
import torch.distributed as dist
import torch_sparse
import numpy as np
from collections import defaultdict
from cagnet.samplers.utils import *

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, \
                                    compute_darts_select_gpu, throw_darts_select_gpu, \
                                    compute_darts1d_gpu, throw_darts1d_gpu, normalize_gpu, \
                                    shift_rowselect_gpu, shift_colselect_gpu, \
                                    scatteri_add_gpu, rowselect_coo_gpu, \
                                    sparse_coo_tensor_gpu


timing = True
baseline_compare = True

def start_time(timer, timing_arg=None):
    if timing_arg is not None:
        start_timing = timing_arg
    else:
        start_timing = timing
    if start_timing:
        timer.record()

def stop_time(start_timer, stop_timer, barrier=False, timing_arg=None):
    if timing_arg is not None:
        start_timing = timing_arg
    else:
        start_timing = timing
    if start_timing:
        stop_timer.record()
        torch.cuda.synchronize()
        time_taken = start_timer.elapsed_time(stop_timer)
        if barrier:
            dist.barrier()
        return time_taken
    else:
        return 0.0

def sage_sampler(adj_matrix, batches, batch_size, frontier_sizes, mb_count_total, n_layers, n_darts_list, \
                        replication, sa_masks, rank, size, row_groups, col_groups,
                        timing_arg, baseline_arg, replicate_graph):

    global timing
    global baseline_compare

    timing = timing_arg
    baseline_compare = baseline_arg

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    start_time(start_timer)
    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    mb_count = batches.size(0)

    rank_c = rank // replication
    rank_col = rank % replication

    # adj_matrices = [[None] * n_layers for x in range(mb_count)] 
    # adj_matrices[i][j] --  mb i layer j
    adj_matrices = [None] * n_layers # adj_matrices[i] --  bulk mb mtx for layer j
    frontiers = [None] * (n_layers + 1) # frontiers[i] -- bulk mb frontiers for layer j

    gpu = torch.device(f"cuda:{torch.cuda.current_device()}")

    batches_expand_rows = torch.arange(mb_count * batch_size, dtype=torch.int32, device=gpu)
    batches_expand_idxs = torch.stack((batches_expand_rows, batches._indices()[1, :]))
    # batches_expand = torch.sparse_coo_tensor(
    #                         batches_expand_idxs,
    #                         batches._values(), 
    #                         size=(mb_count * batch_size, node_count_total))
    print(f"batches_expand_idxs.dtype: {batches_expand_idxs.dtype}", flush=True)
    batches_expand = sparse_coo_tensor_gpu(batches_expand_idxs, batches._values(), 
                                            torch.Size([mb_count * batch_size, node_count_total]))

    if not replicate_graph:
        batches_expand = batches_expand.to_sparse_csr()
    timing_dict["sage-preamble"].append(stop_time(start_timer, stop_timer))

    # adj_matrix = adj_matrix.to_sparse_csr()
    current_frontier = batches_expand

    for i in range(n_layers):
        start_time(total_start_timer)
        # print(f"Sampling layer {i}", flush=True)
        if i == 0:
            nnz = batch_size
        else:
            # Restructure current_frontier to only have 1 nnz/row
            # current_frontier_nnzmask = current_frontier._values().nonzero().squeeze()
            # current_frontier_nnzcols = current_frontier._indices()[1, current_frontier_nnzmask]
            # current_frontier_nnzrows = torch.arange(current_frontier_nnzcols.numel()).cuda()
            start_time(start_timer)
            current_frontier_nnzcols = current_frontier._indices()[1, :]
            current_frontier_nnzrows = torch.arange(current_frontier._nnz()).cuda()
            current_frontier_nnzinds = torch.stack((current_frontier_nnzrows, current_frontier_nnzcols))
            # current_frontier_nnzvals = current_frontier._values()[current_frontier_nnzmask].double()
            #current_frontier_nnzvals = current_frontier._values().double()
            current_frontier_nnzvals = current_frontier._values().float()
            # current_frontier = torch.sparse_coo_tensor(current_frontier_nnzinds, current_frontier_nnzvals,
            #                                             size=torch.Size([current_frontier._nnz(),
            #                                             # size=torch.Size([current_frontier_nnzcols.numel(),
            #                                                                 current_frontier.size(1)]))
            current_frontier = sparse_coo_tensor_gpu(current_frontier_nnzinds, current_frontier_nnzvals, 
                                               torch.Size([current_frontier._nnz(), current_frontier.size(1)]))
            # nnz = current_frontier._nnz() // mb_count
            # nnz = batch_size * (frontier_size ** i)
            nnz = batch_size * int(np.prod(frontier_sizes[:i], dtype=int))
            if not replicate_graph:
                current_frontier = current_frontier.to_sparse_csr()
            timing_dict["sage-startiter"].append(stop_time(start_timer, stop_timer))

        # Expand batches matrix
        p = gen_prob_dist(current_frontier, adj_matrix, mb_count, node_count_total,
                                replication, rank, size, row_groups, col_groups,
                                sa_masks, timing_dict, "sage",
                                timing_arg, replicate_graph)

        # dist.barrier()

        # adj_matrix = adj_matrix.to_sparse_coo()
        # start_time(start_timer)
        # if p.layout == torch.sparse_csr:
        #     p = p.to_sparse_coo()
        # timing_dict["sage-csr2coo"].append(stop_time(start_timer, stop_timer))

        frontier_size = frontier_sizes[i]
        n_darts = n_darts_list[i]
        next_frontier = sample(p, frontier_size, mb_count, node_count_total, n_darts,
                                    replication, rank, size, row_groups, col_groups,
                                    timing_dict, "sage")
        # dist.barrier()

        start_time(start_timer)
        # add explicit 0's to next_frontier to fix the number of rows as bs * sn^l
        next_frontier_nnz = next_frontier._values().nonzero().squeeze()
        frontier_nnz_sizes = torch.histc(next_frontier._indices()[0,next_frontier_nnz], bins=p.size(0))

        frontier_nnz_sizes = torch.clamp(frontier_nnz_sizes, max=frontier_size)
        next_frontier_rows = torch.repeat_interleave(
                                torch.arange(nnz * mb_count, device=gpu),
                                frontier_size)
        nextf_cols_idxs = torch.arange(next_frontier_nnz.size(0), device=gpu)
        frontier_remainder = frontier_size - frontier_nnz_sizes
        ps_f_remain = torch.cumsum(frontier_remainder, dim=0).roll(1)
        ps_f_remain[0] = 0
        nextf_cols_idxs += torch.repeat_interleave(ps_f_remain, frontier_nnz_sizes)
        next_frontier_cols = torch.cuda.LongTensor(next_frontier_rows.size(0)).fill_(0)
        next_frontier_cols.scatter_(0, nextf_cols_idxs, next_frontier._indices()[1,next_frontier_nnz])

        next_frontier_idxs = torch.stack((next_frontier_rows, next_frontier_cols))
        next_frontier_values = torch.cuda.LongTensor(next_frontier_rows.size(0)).fill_(0)
        next_frontier_values[nextf_cols_idxs] = 1

        # Construct sampled adj matrix
        # next_frontier = torch.sparse_coo_tensor(next_frontier_idxs, 
        #                                     next_frontier_values,
        #                                     size=(nnz * mb_count, node_count_total))
        #                                     # size=(batch_size * mb_count, node_count_total))
        next_frontier = sparse_coo_tensor_gpu(next_frontier_idxs, next_frontier_values, 
                                                torch.Size([nnz * mb_count, node_count_total]))

        next_frontier_select = next_frontier._indices()[1,:].view(mb_count * nnz, frontier_size)
        # current_frontier_select = torch.masked_select(current_frontier.col_indices(), \
        #                                         current_frontier.values().bool()).view(current_frontier._nnz(), 1)
        timing_dict["frontier-row-col-select"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        if not replicate_graph:
            current_frontier_select = current_frontier.col_indices().view(current_frontier._nnz(), 1)
        else:
            current_frontier_select = current_frontier._indices()[1,:].view(current_frontier._nnz(), 1)
        if i == 0:
            # frontiers[0] = current_frontier_select.clone()
            frontiers[0] = current_frontier_select

        batch_vals = torch.cuda.LongTensor(current_frontier_select.size()).fill_(1)
        next_frontier_select_vals = next_frontier._values().view(-1)

        batch_rows = torch.arange(mb_count * nnz, device=gpu).view(mb_count * nnz, 1)
        next_frontier_select_rows = next_frontier._indices()[0,:].view(-1)

        nnz_mask = next_frontier_select_vals.nonzero().squeeze()
        adj_sample_rows = next_frontier_select_rows[nnz_mask]
        adj_sample_cols = torch.arange(next_frontier_select.numel(), device=gpu)
        adj_sample_cols = adj_sample_cols.remainder(next_frontier_select.size(1) * nnz)
        adj_sample_cols = adj_sample_cols[nnz_mask]
        adj_matrices_indices = torch.stack((adj_sample_rows, adj_sample_cols))
        # adj_matrices_values = torch.cuda.DoubleTensor(adj_sample_rows.size(0)).fill_(1)
        adj_matrices_values = torch.cuda.FloatTensor(adj_sample_rows.size(0)).fill_(1)

        # adj_matrix_sample = torch.sparse_coo_tensor(adj_matrices_indices, adj_matrices_values, 
        #                         size=torch.Size([mb_count * nnz, next_frontier_select.size(1) * nnz]))
        adj_matrix_sample = sparse_coo_tensor_gpu(adj_matrices_indices, adj_matrices_values, 
                                            torch.Size([mb_count * nnz, next_frontier_select.size(1) * nnz]))
        # adj_matrices[i] = adj_matrix_sample
        # frontiers[i + 1] = next_frontier_select.clone()
        if i + 1 == n_layers:
            frontiers[i + 1] = next_frontier_select
        # else:
        #     del next_frontier_select
        # del p

        # if i > 0:
        #     del current_frontier_select
        current_frontier = next_frontier
        timing_dict["adj-row-col-select"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        adj_matrices[i] = adj_matrix_sample.to_sparse_csr()
        timing_dict["adj-coo2csr"].append(stop_time(start_timer, stop_timer))
        # del next_frontier_select
        # del current_frontier_select
        timing_dict["sage-samplingiter"].append(stop_time(total_start_timer, total_stop_timer))

    if timing:
        for k, v in sorted(timing_dict.items()):
            if (k.startswith("spgemm") and k != "spgemm-misc") or k == "probability-spgemm" or k == "row-select-spgemm" or k == "col-select-spgemm" or k == "sampling-iters" or k == "frontier-row-col-select" or k == "adj-row-col-select" or k.startswith("sample") or k == "compute-p" or k == "sage-startiter" or k == "sage-csr2coo" or k == "sage-preamble" or k == "sage-samplingiter":
                v_tens = torch.cuda.FloatTensor(1).fill_(sum(v))
                v_tens_recv = []
                for i in range(size):
                    v_tens_recv.append(torch.cuda.FloatTensor(1).fill_(0))
                dist.all_gather(v_tens_recv, v_tens)

                if rank == 0:
                    min_time = min(v_tens_recv).item()
                    max_time = max(v_tens_recv).item()
                    avg_time = sum(v_tens_recv).item() / size
                    med_time = sorted(v_tens_recv)[size // 2].item()

                    print(f"{k} min: {min_time} max: {max_time} avg: {avg_time} med: {med_time}")
            dist.barrier()
        for k, v in timing_dict.items():
            if len(v) > 0:
                avg_time = sum(v) / len(v)
            else:
                avg_time = -1.0
            print(f"{k} total_time: {sum(v)} avg_time {avg_time} len: {len(v)}")
    # return current_frontier_select, next_frontier_select, adj_matrices
    return frontiers, adj_matrices
