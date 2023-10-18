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
                                    scatteri_add_gpu, rowselect_coo_gpu


def start_time(timer):
    timer.record()

def stop_time(start_timer, stop_timer, barrier=False):
    stop_timer.record()
    torch.cuda.synchronize()
    time_taken = start_timer.elapsed_time(stop_timer)
    if barrier:
        dist.barrier()
    return time_taken
    
def ladies_sampler(adj_matrix, batches, batch_size, frontier_size, mb_count_total, n_layers, n_darts, \
                        semibulk_size, replication, sa_masks, rank, size, \
                        row_groups, col_groups, timing, replicate_graph):

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    frontier_size = frontier_size[0] # one layer
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

    # adj_matrices = [[None] * n_layers for x in range(mb_count)] # adj_matrices[i][j] --  mb i layer j
    adj_matrices = [None] * n_layers # adj_matrices[i] --  bulk minibatch matrix for layer j

    start_time(total_start_timer)
    for i in range(n_layers):
        if i == 0:
            nnz = batch_size
        else:
            nnz = current_frontier[0, :].size(0)

        batches = batches.to_sparse_csr()
        # adj_matrix_sq = adj_matrix.square().to_sparse_csr()
        adj_matrix_sq = adj_matrix.values().square()
        adj_matrix_sq = torch.sparse_csr_tensor(adj_matrix.crow_indices(), adj_matrix.col_indices(),
                                                    adj_matrix_sq, adj_matrix.size())
        p = gen_prob_dist(batches, adj_matrix_sq, mb_count, node_count_total, replication, rank, size, \
                            row_groups, col_groups, sa_masks, timing_dict, "ladies", timing, replicate_graph)

        next_frontier = sample(p, frontier_size, mb_count, node_count_total, n_darts, replication, rank, size, \
                                    row_groups, col_groups, timing_dict, "ladies")
        # adj_matrix = adj_matrix.to_sparse_coo()
        if p.layout == torch.sparse_csr:
            p = p.to_sparse_coo()
        batches = batches.to_sparse_coo()

        batches_select, next_frontier_select, adj_matrix_sample = \
                    select(next_frontier, adj_matrix_sq, batches, sa_masks, nnz, \
                                    batch_size, frontier_size, mb_count, mb_count_total, node_count_total, \
                                    replication, rank, size, row_groups, col_groups, timing_dict, i, "ladies",
                                    semibulk_size)

        # adj_matrices[i] = adj_matrix_sample
        adj_matrices[i] = adj_matrix_sample.to_sparse_csr()
        current_frontier = next_frontier


    print(f"total_time: {stop_time(total_start_timer, total_stop_timer)}", flush=True)
    if timing:
        for k, v in sorted(timing_dict.items()):
            if (k.startswith("spgemm") and k != "spgemm-misc") or k == "probability-spgemm" or k == "row-select-spgemm" or k == "col-select-spgemm" or k == "sampling-iters":
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
