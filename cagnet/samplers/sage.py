import math
import torch
import torch.distributed as dist
import torch_sparse
from collections import defaultdict
from cagnet.samplers.utils import *

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, \
                                    compute_darts_select_gpu, throw_darts_select_gpu, \
                                    compute_darts1d_gpu, throw_darts1d_gpu, normalize_gpu, \
                                    shift_rowselect_gpu, shift_colselect_gpu, \
                                    scatterd_add_gpu, scatteri_add_gpu, rowselect_coo_gpu, \
                                    sparse_coo_tensor_gpu


timing = True
baseline_compare = False

def start_time(timer):
    if timing == True:
        timer.record()

def stop_time(start_timer, stop_timer):
    if timing == True:
        stop_timer.record()
        torch.cuda.synchronize()
        return start_timer.elapsed_time(stop_timer)
    else:
        return 0.0

def sage_sampler(adj_matrix, batches, batch_size, frontier_size, mb_count_total, n_layers, n_darts, \
                        replication, sa_masks, sa_recv_buff, rank, size, row_groups, col_groups):

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    current_frontier = torch.cuda.IntTensor(mb_count_total, batch_size + frontier_size)

    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    mb_count = batches.size(0)

    rank_c = rank // replication
    rank_col = rank % replication

    n_darts_col = n_darts // replication
    if rank_col == replication - 1:
        n_darts_col = n_darts - (replication - 1) * n_darts_col
    n_darts_col = n_darts

    # adj_matrices = [[None] * n_layers for x in range(mb_count)] 
    # adj_matrices[i][j] --  mb i layer j
    adj_matrices = [None] * n_layers # adj_matrices[i] --  bulk mb mtx for layer j

    if not baseline_compare:
        start_time(total_start_timer)
    for i in range(n_layers):
        if i == 0:
            nnz = batch_size
        else:
            nnz = current_frontier[0, :].size(0)

        # Expand batches matrix
        if baseline_compare:
            total_start_timer.record()
        batches_expand_rows = torch.arange(mb_count * nnz, \
                                                device=torch.device("cuda:0"))
        batches_expand_idxs = torch.stack(
                                (batches_expand_rows, batches._indices()[1, :])
                                )
        batches_expand = torch.sparse_coo_tensor(
                                batches_expand_idxs,
                                batches._values(), 
                                size=(mb_count * nnz, node_count_total))

        p = gen_prob_dist(batches_expand, adj_matrix, mb_count, node_count_total,
                                replication, rank, size, row_groups, col_groups,
                                sa_masks, sa_recv_buff, timing_dict, "sage")

        next_frontier = sample(p, frontier_size, mb_count, node_count_total, n_darts,
                                    replication, rank, size, row_groups, col_groups,
                                    timing_dict, "sage")

        if baseline_compare:
            total_stop_timer.record()
            torch.cuda.synchronize()
            total_time = total_start_timer.elapsed_time(total_stop_timer)
            print(f"total_time: {total_time}", flush=True)

            batches_select = None
            next_frontier_select = None
            adj_matrices = None
            break # for comparing with quiver

        # add explicit 0's to next_frontier
        next_frontier_nnz = next_frontier._values().nonzero().squeeze()
        frontier_nnz_sizes = torch.histc(next_frontier._indices()[0,next_frontier_nnz], bins=p.size(0))

        frontier_nnz_sizes = torch.clamp(frontier_nnz_sizes, max=frontier_size)
        next_frontier_rows = torch.repeat_interleave(
                                torch.arange(batch_size * mb_count, device=torch.device("cuda:0")),
                                frontier_size)
        nextf_cols_idxs = torch.arange(next_frontier_nnz.size(0), device=torch.device("cuda:0"))
        frontier_remainder = frontier_size - frontier_nnz_sizes
        ps_f_remain = torch.cumsum(frontier_remainder, dim=0).roll(1)
        ps_f_remain[0] = 0
        nextf_cols_idxs += torch.repeat_interleave(ps_f_remain, frontier_nnz_sizes)
        next_frontier_cols = torch.cuda.LongTensor(next_frontier_rows.size(0)).fill_(0)
        next_frontier_cols.scatter_(0, nextf_cols_idxs, next_frontier._indices()[1,next_frontier_nnz])

        next_frontier_idxs = torch.stack((next_frontier_rows, next_frontier_cols))
        next_frontier_values = torch.cuda.LongTensor(next_frontier_rows.size(0)).fill_(0)
        next_frontier_values[nextf_cols_idxs] = 1

        next_frontier = torch.sparse_coo_tensor(next_frontier_idxs, 
                                            next_frontier_values,
                                            size=(batch_size * mb_count, node_count_total))

        # next_frontier = next_frontier.coalesce()
        # next_frontier._values().fill_(1)

        batches_select, next_frontier_select, adj_matrix_sample = \
                    select(next_frontier, adj_matrix, batches, sa_masks, sa_recv_buff, 
                                nnz, batch_size, frontier_size, mb_count, 
                                mb_count_total, node_count_total, replication, rank, 
                                size, row_groups, col_groups, timing_dict, i, "sage")

        adj_matrices[i] = adj_matrix_sample
        current_frontier = next_frontier

    print(f"total_time: {stop_time(total_start_timer, total_stop_timer)}", flush=True)
    for k, v in sorted(timing_dict.items()):
        if (k.startswith("spgemm") and k != "spgemm-misc") or k == "probability-spgemm" or k == "row-select-spgemm" or k == "col-select-spgemm":
            # print(f"{k} times: {v}")
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
    return batches_select, next_frontier_select, adj_matrices
