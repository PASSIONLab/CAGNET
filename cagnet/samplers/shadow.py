import math
import torch
import torch.distributed as dist
import torch_sparse
import numpy as np
from collections import defaultdict
from cagnet.samplers.utils import *
from sparse_coo_tensor_cpp import *

timing = True

# def start_time(timer, timing_arg=None):
#     if timing_arg is not None:
#         start_timing = timing_arg
#     else:
#         start_timing = timing
#     if start_timing:
#         timer.record()
# 
# def stop_time(start_timer, stop_timer, stream=None, barrier=False, timing_arg=None):
#     if timing_arg is not None:
#         start_timing = timing_arg
#     else:
#         start_timing = timing
#     if start_timing:
#         stop_timer.record()
#         if stream is None:
#             torch.cuda.synchronize()
#         else:
#             stream.synchronize()
#         time_taken = start_timer.elapsed_time(stop_timer)
#         if barrier:
#             dist.barrier()
#         return time_taken
#     else:
#         return 0.0

def shadow_sampler(adj_matrix, batches, batch_size, frontier_sizes, mb_count_total, n_layers, n_darts_list, \
                        replication, rank, size, row_groups, col_groups,
                        timing_arg, timing_dict, replicate_graph, epoch):

    global timing

    timing = timing_arg

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict_inner = defaultdict(list)

    rank_c = rank // replication
    rank_col = rank % replication
    mb_count = batches.size(0)
    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    gpu = torch.device(f"cuda:{torch.cuda.current_device()}")

    # adj_matrix = adj_matrix.t().to_sparse_csr()

    batches_expand_rows = torch.arange(mb_count * batch_size, dtype=torch.int32, device=gpu)
    batches_expand_idxs = torch.stack((batches_expand_rows, batches._indices()[1, :]))
    batches_expand = sparse_coo_tensor_gpu(batches_expand_idxs, batches._values(), 
                                            torch.Size([mb_count * batch_size, node_count_total]))


    current_frontier = batches_expand

    # frontiers = torch.cuda.LongTensor(mb_count * batch_size, node_count_total).fill_(0) # TODO: make sparse 
    # frontiers_idxs = batches._indices()[1,:].unsqueeze(1)
    # frontiers.scatter_(1, frontiers_idxs, 1) 
    frontiers_rows = torch.arange(mb_count * batch_size, dtype=torch.int64).to(gpu)
    frontiers_cols = batches._indices()[1,:]
    frontiers_idxs = torch.stack((frontiers_rows, frontiers_cols))
    frontiers_vals = torch.cuda.LongTensor(mb_count * batch_size).fill_(1)
    frontiers = torch.sparse_coo_tensor(frontiers_idxs, frontiers_vals, 
                                            size=torch.Size([mb_count * batch_size, node_count_total]))
    current_frontier_mbids = None
    if epoch >= 1:
        start_time(start_timer)
    for i in range(n_layers):
        neighbors = batch_size * int(np.prod(frontier_sizes[:(i + 1)], dtype=int))

        p = gen_prob_dist(current_frontier, adj_matrix, mb_count, node_count_total,
                                replication, rank, size, row_groups, col_groups,
                                None, timing_dict_inner, "sage", timing_arg, replicate_graph)

        # frontier_size = frontier_sizes[i]
        frontier_size = frontier_sizes[0]
        n_darts = n_darts_list[i]
        next_frontier = sample(p, frontier_size, mb_count, node_count_total, n_darts,
                                    replication, rank, size, row_groups, col_groups,
                                    timing_dict_inner, "sage")

        # collapse next_frontier to mb_count x node_count_total sparse matrix
        # TODO: Change this when collapsed frontier is sparse
        next_frontier_mask = next_frontier._values().nonzero().squeeze()
        # collapsed_frontier_rows = next_frontier._indices()[0, next_frontier_mask]

        if i == 0:
            collapsed_frontier_rows = next_frontier._indices()[0, next_frontier_mask]
        else:
            collapsed_frontier_rows = current_frontier_mbids[next_frontier._indices()[0, next_frontier_mask]]

        # collapsed_frontier_rows = collapsed_frontier_rows.div(row_div, rounding_mode="floor")
        collapsed_frontier_cols = next_frontier._indices()[1, next_frontier_mask]
        collapsed_frontier_idxs = torch.stack((collapsed_frontier_rows, collapsed_frontier_cols))
        if collapsed_frontier_rows.numel() == 1 and collapsed_frontier_cols.numel() == 1:
            collapsed_frontier_idxs = collapsed_frontier_idxs.reshape(2, 1)
        collapsed_frontier_vals = torch.cuda.LongTensor(collapsed_frontier_idxs.size(1)).fill_(1)

        collapsed_frontier = torch.sparse_coo_tensor(collapsed_frontier_idxs, 
                                                        collapsed_frontier_vals,
                                                        torch.Size([mb_count * batch_size, node_count_total]))
        collapsed_frontier = collapsed_frontier.coalesce()
        # collapsed_frontier_dense = collapsed_frontier.to_dense()
        # frontiers = frontiers + collapsed_frontier_dense
        current_frontier_mbids = collapsed_frontier_idxs[0,:]
        frontiers = frontiers + collapsed_frontier
        frontiers = frontiers.coalesce()

        if i < n_layers - 1:
            # expand_frontier_rows = torch.arange(collapsed_frontier._nnz()).to(gpu)
            if collapsed_frontier_cols.dim() == 0:
                collapsed_frontier_cols = collapsed_frontier_cols.unsqueeze(0)
            expand_frontier_rows = torch.arange(collapsed_frontier_cols.size(0)).to(gpu)
            expand_frontier_cols = collapsed_frontier_cols
            expand_frontier_vals = collapsed_frontier_vals
            expand_frontier_idxs = torch.stack((expand_frontier_rows, expand_frontier_cols))
            expand_frontier = torch.sparse_coo_tensor(expand_frontier_idxs, 
                                                            expand_frontier_vals,
                                                            torch.Size([mb_count * neighbors, node_count_total]))
            current_frontier = expand_frontier
    if epoch >= 1:
        timing_dict["shadow-sampling"].append(stop_time(start_timer, stop_timer))

    if epoch >= 1:
        start_time(start_timer)
    # sampled_frontiers = frontiers.nonzero()[:,1]
    sampled_frontiers = frontiers._indices()[1,:]

    batch_ids = torch.div(frontiers._indices()[0,:], batch_size, rounding_mode="floor").int()
    batch_sizes = torch.histc(batch_ids, bins=mb_count, min=0, max=mb_count-1)
    ps_batch_sizes = torch.cumsum(batch_sizes, 0).roll(1)
    ps_batch_sizes[0] = 0

    row_lengths = adj_matrix.crow_indices()[1:] - adj_matrix.crow_indices()[:-1]
    rowselect_adj_crows = torch.cuda.LongTensor(sampled_frontiers.size(0) + 1).fill_(0)
    rowselect_adj_crows[1:] = torch.cumsum(row_lengths[sampled_frontiers], dim=0)
    rowselect_adj_crows[0] = 0
    rowselect_adj_nnz = rowselect_adj_crows[-1].item()
    rowselect_adj_cols = torch.cuda.LongTensor(rowselect_adj_nnz).fill_(0)
    rowselect_adj_vals = torch.cuda.LongTensor(rowselect_adj_nnz).fill_(0)
    rowselect_csr_dupes_gpu(sampled_frontiers, adj_matrix.crow_indices().long(), adj_matrix.col_indices(),
                                adj_matrix.values(), rowselect_adj_crows, rowselect_adj_cols, rowselect_adj_vals, 
                                sampled_frontiers.size(0), adj_matrix._nnz())

    sampled_frontier_size = sampled_frontiers.size(0)
    row_select_adj = torch.sparse_csr_tensor(rowselect_adj_crows, rowselect_adj_cols, rowselect_adj_vals,
                                                torch.Size([sampled_frontier_size, node_count_total]))

    # sampled_frontiers_rowids = frontiers.nonzero()[:,0]
    sampled_frontiers_rowids = frontiers._indices()[0,:]
    sampled_frontiers_csr = frontiers.to_sparse_csr()

    colselect_idxs = torch.cuda.LongTensor(row_select_adj._nnz()).fill_(-1)
    shadow_colselect_gpu(sampled_frontiers, sampled_frontiers_rowids, sampled_frontiers_csr.crow_indices(),
                            sampled_frontiers_csr.col_indices(), row_select_adj.crow_indices().long(), 
                            row_select_adj.col_indices(), colselect_idxs, ps_batch_sizes, batch_size, 
                            sampled_frontiers.size(0), row_select_adj._nnz())

    colselect_mask = colselect_idxs > -1
    sampled_adjs_cols = row_select_adj.col_indices()[colselect_mask]
    # for i in range(sampled_frontier_size):
    #     vtx = sampled_frontiers[i]
    #     frontiers_col_indices = frontiers._indices()[1, frontiers._indices()[0,:] == i]
    #     adj_col_indices = sampled_adjs_cols[row_select_adj.to_sparse_coo()._indices()[0, colselect_mask] == i]
    #     print(f"i: {i} vtx: {vtx} sampled_frontiers[i]: {frontiers_col_indices} sampled_adjs[i]: {adj_col_indices}", flush=True)
    sampled_adjs_cols = colselect_idxs[colselect_mask]
    sampled_adjs_vals = row_select_adj.values()[colselect_mask]

    sampled_adjs_rows = row_select_adj.to_sparse_coo()._indices()[0,:]
    sampled_adjs_rows = sampled_adjs_rows[colselect_mask]

    sampled_adj_indices = torch.stack((sampled_adjs_rows, sampled_adjs_cols))
    sampled_frontier_size = sampled_frontiers.size(0)
    sampled_adjs = torch.sparse_coo_tensor(sampled_adj_indices, sampled_adjs_vals, 
                                                torch.Size([sampled_frontier_size, sampled_frontier_size]))

    # sampled_adjs = sampled_adjs.t()

    sampled_frontiers_split = torch.split(sampled_frontiers, batch_sizes.tolist())
    if epoch >= 1:
        timing_dict["shadow-selection"].append(stop_time(start_timer, stop_timer))

    # sampled_adjs_indices_split = torch.split(sampled_adjs._indices(), batch_sizes.tolist(), dim=1)
    # sampled_adjs_vals_split = torch.split(sampled_adjs._values(), batch_sizes.tolist())
    if epoch >= 1:
        start_time(start_timer)
    sampled_adjs_split = []
    for i in range(mb_count):
        if i < mb_count - 1:
            sampled_adjs_mask = (sampled_adjs._indices()[0,:] >= ps_batch_sizes[i]) & \
                                    (sampled_adjs._indices()[0,:] < ps_batch_sizes[i + 1])
        else:
            sampled_adjs_mask = (sampled_adjs._indices()[0,:] >= ps_batch_sizes[i]) & \
                                    (sampled_adjs._indices()[0,:] < batch_sizes.sum().item())
        sampled_adjs_indices_split = sampled_adjs._indices()[:, sampled_adjs_mask]
        # sampled_adjs_indices_split[0,:] -= sampled_adjs_indices_split[0,:].min().item()
        sampled_adjs_indices_split[0,:] -= ps_batch_sizes[i]
        sampled_adjs_values_split = sampled_adjs._values()[sampled_adjs_mask]
        sampled_adjs_split.append(torch.sparse_coo_tensor(sampled_adjs_indices_split,
                                                            sampled_adjs_values_split,
                                                            torch.Size([batch_sizes[i], batch_sizes[i]])).t())
    if epoch >= 1:
        timing_dict["shadow-extraction"].append(stop_time(start_timer, stop_timer))

    # if timing:
    #     # for k, v in sorted(timing_dict.items()):
    #     #     if (k.startswith("spgemm") and k != "spgemm-misc") or k == "probability-spgemm" or k == "row-select-spgemm" or k == "col-select-spgemm" or k == "sampling-iters" or k == "frontier-row-col-select" or k == "adj-row-col-select" or k.startswith("sample") or k == "compute-p" or k == "sage-startiter" or k == "sage-csr2coo" or k == "sage-preamble" or k == "sage-samplingiter":
    #     #         v_tens = torch.cuda.FloatTensor(1).fill_(sum(v))
    #     #         v_tens_recv = []
    #     #         for i in range(size):
    #     #             v_tens_recv.append(torch.cuda.FloatTensor(1).fill_(0))
    #     #         dist.all_gather(v_tens_recv, v_tens)

    #     #         if rank == 0:
    #     #             min_time = min(v_tens_recv).item()
    #     #             max_time = max(v_tens_recv).item()
    #     #             avg_time = sum(v_tens_recv).item() / size
    #     #             med_time = sorted(v_tens_recv)[size // 2].item()

    #     #             print(f"{k} min: {min_time} max: {max_time} avg: {avg_time} med: {med_time}")
    #     #     dist.barrier()
    #     for k, v in timing_dict.items():
    #         if len(v) > 0:
    #             avg_time = sum(v) / len(v)
    #         else:
    #             avg_time = -1.0
    #         print(f"{k} total_time: {sum(v)} avg_time {avg_time} len: {len(v)}")
    # # return sampled_frontiers, sampled_adjs
    return sampled_frontiers_split, sampled_adjs_split
