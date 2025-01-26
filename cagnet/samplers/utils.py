import math 
import numpy as np
import torch 
import torch.distributed as dist 
import torch_geometric
from torch_geometric.data import Data, Dataset
import torch_sparse 
from collections import defaultdict
from pathlib import Path
import os
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from matplotlib import pyplot as plt
from atlasify import atlasify
import seaborn as sns
import yaml

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, \
                                    compute_darts_select_gpu, throw_darts_select_gpu, \
                                    compute_darts1d_gpu, throw_darts1d_gpu, normalize_gpu, \
                                    normalize_csr_gpu, shift_rowselect_gpu, shift_colselect_gpu, \
                                    scatteri_add_gpu, rowselect_coo_gpu, \
                                    rowselect_csr_gpu, sparse_coo_tensor_gpu, spgemm_gpu, coogeam_gpu, \
                                    sparse_csr_tensor_gpu, nsparse_spgemm, rearrange_rows_gpu, \
                                    rearrangel_rows_gpu, reduce_sum_gpu, sum_csrd_gpu, sum_csri_gpu


timing = True

def start_time(timer, timing_arg=None):
    if timing_arg is not None:
        start_timing = timing_arg
    else:
        start_timing = timing
    if start_timing:
        timer.record()

def stop_time(start_timer, stop_timer, stream=None, barrier=False, timing_arg=None):
    if timing_arg is not None:
        start_timing = timing_arg
    else:
        start_timing = timing
    if start_timing:
        stop_timer.record()
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        time_taken = start_timer.elapsed_time(stop_timer)
        if barrier:
            dist.barrier(torch.distributed.group.WORLD)
        return time_taken
    else:
        return 0.0

def stop_time_add(start_timer, stop_timer, timing_dict, range_name, barrier=False):
    if timing_dict is not None:
        timing_dict[range_name].append(stop_time(start_timer, stop_timer, barrier))

def csr_allreduce(mat, left, right, rank, name=None, alg=None, timing_dict=None):
    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    while left < right:
        start_time(start_timer)
        group_size = right - left + 1
        mid_rank = (left + right) // 2

        if rank <= mid_rank:
            recv_rank = rank + (group_size // 2)
        else:
            recv_rank = rank - (group_size // 2)

        ops = [None, None]
        nnz_send = torch.cuda.IntTensor(1).fill_(mat._nnz())
        nnz_recv = torch.cuda.IntTensor(1)
        
        ops[0] = dist.P2POp(dist.isend, nnz_send, recv_rank)
        ops[1] = dist.P2POp(dist.irecv, nnz_recv, recv_rank)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait() 

        # torch.cuda.synchronize()
        rows_recv = torch.cuda.IntTensor(mat.size(0) + 1)
        cols_recv = torch.cuda.IntTensor(nnz_recv.item())
        if mat.dtype == torch.float32:
            vals_recv = torch.cuda.FloatTensor(nnz_recv.item())
        elif mat.dtype == torch.float64:
            vals_recv = torch.cuda.DoubleTensor(nnz_recv.item())

        ops = [None] * 6
        ops[0] = dist.P2POp(dist.isend, mat.crow_indices().int(), recv_rank, tag=0)
        ops[1] = dist.P2POp(dist.isend, mat.col_indices().int(), recv_rank, tag=1)
        ops[2] = dist.P2POp(dist.isend, mat.values(), recv_rank, tag=2)
        ops[3] = dist.P2POp(dist.irecv, rows_recv, recv_rank, tag=0)
        ops[4] = dist.P2POp(dist.irecv, cols_recv, recv_rank, tag=1)
        ops[5] = dist.P2POp(dist.irecv, vals_recv, recv_rank, tag=2)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait() 
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-comm-{name}")

        # torch.cuda.synchronize()
        start_time(start_timer)
        torch.cuda.synchronize()
        mat_recv = torch.sparse_csr_tensor(rows_recv.long(), cols_recv.long(), vals_recv, size=mat.size())
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-csrinst-{name}")
        start_time(start_timer)

        if True or alg == "ladies":
            mat = mat + mat_recv
        else:
            # Add mat + mat_recv
            mat_row_lens = mat.crow_indices()[1:] - mat.crow_indices()[:-1]
            mat_recv_row_lens = mat_recv.crow_indices()[1:] - mat_recv.crow_indices()[:-1]

            mat_sum_row_lens = mat_row_lens + mat_recv_row_lens
            mat_sum_crows = torch.cuda.LongTensor(mat_sum_row_lens.size(0) + 1).fill_(0)
            mat_sum_crows[1:] = torch.cumsum(mat_sum_row_lens, dim=0)
            mat_sum_crows[0] = 0
            mat_sum_cols = torch.cuda.LongTensor(mat._nnz() + mat_recv._nnz()).fill_(0)
            mat_row_lens_nnzrows = mat_row_lens.nonzero().squeeze()
            mat_recv_row_lens_nnzrows = mat_recv_row_lens.nonzero().squeeze()
            reduce_sum_gpu(mat_sum_crows, mat.crow_indices(), mat_recv.crow_indices(), 
                            mat_sum_cols, mat.col_indices(), mat_recv.col_indices(),
                            mat_row_lens_nnzrows, mat_recv_row_lens_nnzrows)

            mat_sum_vals = torch.cuda.FloatTensor(mat_sum_cols.size(0)).fill_(1.0)
            mat_sum_crows = mat_sum_crows.cpu()
            mat_sum_cols = mat_sum_cols.cpu()
            mat_sum_vals = mat_sum_vals.cpu()
            mat = torch.sparse_csr_tensor(mat_sum_crows, mat_sum_cols, mat_sum_vals, mat.size())
            mat = mat.to(device)
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-sum-{name}")

        if rank <= mid_rank:
            right = mid_rank
        else:
            left = mid_rank + 1
    return mat

def col_select15D(sample_mtx, col_select_mtx, bulk_items, node_count_total, bulk_size, semibulk_size, 
                        replication, rank, size, row_groups, col_groups, name, alg, timing_dict=None):

    rank_c = rank // replication
    rank_col = rank % replication
    stages = (bulk_items // semibulk_size) // replication
   
    sample_mtxs = [None] * stages

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_inner_timer = torch.cuda.Event(enable_timing=True)
    stop_inner_timer = torch.cuda.Event(enable_timing=True)

    if alg == "sage":
        mask = torch.cuda.BoolTensor(sample_mtx.size(1)).fill_(0)
        col_indices_mask = torch.cuda.LongTensor(sample_mtx.size(1)).fill_(0)
        idx_range = torch.arange(sample_mtx.size(1)).cuda()
    
    start_time(start_timer)
    for i in range(stages): 
        q = stages * rank_col + i 

        # Extract semibulk_size batches from sample_mtx
        start_time(start_inner_timer)
        batch_start = q * bulk_size * semibulk_size
        batch_stop = (q + 1) * bulk_size * semibulk_size
        batch_crow_indices = sample_mtx.crow_indices()[batch_start:(batch_stop + 1)].clone()
        batch_crow_indices -= batch_crow_indices[0].item()
        start_nnz = sample_mtx.crow_indices()[batch_start]
        stop_nnz = sample_mtx.crow_indices()[batch_stop]

        batch_cols = sample_mtx.col_indices()[start_nnz:stop_nnz]
        batch_vals = sample_mtx.values()[start_nnz:stop_nnz]

        batch_mtx = torch.sparse_csr_tensor(batch_crow_indices, batch_cols, batch_vals, \
                                                torch.Size([bulk_size * semibulk_size, sample_mtx.size(1)]))

        batch_mtx = batch_mtx.to_sparse_coo()
        stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-batch-{name}", barrier=True)

        start_time(start_inner_timer)
        # Extract semibulk_size column extraction matrices
        chunk_row_start = q * semibulk_size * node_count_total
        chunk_row_stop = (q + 1) * semibulk_size * node_count_total
        chunk_row_mask = (col_select_mtx._indices()[0, :] >= chunk_row_start) & \
                         (col_select_mtx._indices()[0, :] < chunk_row_stop)
        col_chunk_indices = col_select_mtx._indices()[:, chunk_row_mask]
        col_chunk_indices[0,:] -= chunk_row_start
        col_chunk_values = col_select_mtx._values()[chunk_row_mask]
        stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-colsel-{name}", barrier=True)

        start_time(start_inner_timer)
        if alg == "ladies":
            sample_chunk_indices, sample_chunk_values = torch_sparse.spspmm(batch_mtx._indices(), \
                                                            batch_mtx._values().double(), col_chunk_indices, \
                                                            col_chunk_values, batch_mtx.size(0), \
                                                            semibulk_size * node_count_total, \
                                                            # col_select_mtx.size(1))
                                                            col_select_mtx.size(1), coalesced=True)
        elif alg == "sage":
            mask.fill_(0)
            batch_mtx = batch_mtx.to_sparse_coo()
            mask[col_chunk_indices[0,:]] = True
            col_mask = torch.gather(mask, 0, batch_mtx._indices()[1,:])

            sample_rows = batch_mtx._indices()[0, col_mask]
            sample_cols = batch_mtx._indices()[0, col_mask]
            col_indices_mask.fill_(0)
            col_indices_mask[col_chunk_indices[0,:]] = idx_range[:col_chunk_indices.size(1)].long()
            sample_cols_scaled = torch.gather(col_indices_mask, 0, sample_cols)
            sample_chunk_values = batch_mtx._values()[col_mask]
            sample_chunk_indices = torch.stack((sample_rows, sample_cols_scaled))
        stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-spgemm-{name}", barrier=True)
        sample_mtxs[i] = torch.sparse_coo_tensor(sample_chunk_indices, sample_chunk_values,
                                                size=torch.Size([batch_mtx.size(0), col_select_mtx.size(1)]))

    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-extract-iters-{name}", barrier=True)

    start_time(start_timer)
    sample_mtx_int = torch.cat(sample_mtxs, dim=0)
    row_start = (bulk_items // replication) * bulk_size * rank_col
    sample_mtx_rows = sample_mtx_int._indices()[0,:] + row_start
    sample_mtx_indices = torch.stack((sample_mtx_rows, sample_mtx_int._indices()[1,:]))
    sample_mtx_int = torch.sparse_coo_tensor(sample_mtx_indices, sample_mtx_int._values(), \
                        size=torch.Size([bulk_items * bulk_size, col_select_mtx.size(1)])).to_sparse_csr()


    rank_row_start = rank_c * replication
    rank_row_stop = (rank_c + 1) * replication - 1
    sample_mtx_int = sample_mtx_int.to_sparse_csr()
    sample_mtx_red = csr_allreduce(sample_mtx_int, rank_row_start, rank_row_stop, rank)
    sample_mtx_red = sample_mtx_red.to_sparse_coo()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-extract-reduce-{name}", barrier=True)

    return sample_mtx_red._indices(), sample_mtx_red._values()

# def col_select15D(sample_mtx, next_frontier_select, mb_count, batch_size, replication, rank, size, \
#                         row_groups, col_groups, name, timing_dict=None):
# 
#     rank_c = rank // replication
#     rank_col = rank % replication
#     stages = mb_count // replication
#    
#     sample_mtxs = [None] * stages
# 
#     start_timer = torch.cuda.Event(enable_timing=True)
#     stop_timer = torch.cuda.Event(enable_timing=True)
# 
#     start_inner_timer = torch.cuda.Event(enable_timing=True)
#     stop_inner_timer = torch.cuda.Event(enable_timing=True)
# 
#     mask = torch.cuda.BoolTensor(sample_mtx.size(1)).fill_(0)
#     col_indices_mask = torch.cuda.LongTensor(sample_mtx.size(1)).fill_(0)
#     idx_range = torch.arange(sample_mtx.size(1)).long().cuda()
#     start_time(start_timer)
#     for i in range(stages): 
#         q = stages * rank_col + i 
# 
#         start_time(start_inner_timer)
#         next_frontier_unique = next_frontier_select[q].unique()
#         batch_crow_indices = sample_mtx.crow_indices()[(q * batch_size):((q + 1) * batch_size + 1)].clone()
#         batch_crow_indices -= batch_crow_indices[0].item()
#         start_nnz = sample_mtx.crow_indices()[q * batch_size]
#         stop_nnz = sample_mtx.crow_indices()[(q + 1) * batch_size]
# 
#         batch_cols = sample_mtx.col_indices()[start_nnz:stop_nnz]
#         batch_vals = sample_mtx.values()[start_nnz:stop_nnz]
# 
#         batch_mtx = torch.sparse_csr_tensor(batch_crow_indices, batch_cols, batch_vals, \
#                                                     torch.Size([batch_size, sample_mtx.size(1)]))
#         stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-batch-{name}", barrier=True)
# 
#         start_time(start_inner_timer)
#         mask.fill_(0)
#         batch_mtx = batch_mtx.to_sparse_coo()
#         mask[next_frontier_unique] = True
#         col_mask = torch.gather(mask, 0, batch_mtx._indices()[1,:])
#         stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-gather-{name}", barrier=True)
# 
#         start_time(start_inner_timer)
#         sample_rows = batch_mtx._indices()[0, col_mask]
#         sample_cols = batch_mtx._indices()[1, col_mask] 
#         col_indices_mask.fill_(0)
#         col_indices_mask[next_frontier_unique] = idx_range[:next_frontier_unique.size(0)]
#         sample_cols_scaled = torch.gather(col_indices_mask, 0, sample_cols)
#         sample_vals = batch_mtx._values()[col_mask]
#         sample_indices = torch.stack((sample_rows, sample_cols_scaled))
#         sample_mtxs[i] = torch.sparse_coo_tensor(sample_indices, sample_vals,
#                                                     size=torch.Size([batch_size, next_frontier_select.size(1)]))
#         stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-cols-{name}", barrier=True)
# 
#         # start_time(start_inner_timer)
#         # batch_mtx = batch_mtx.t()
#         # stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-transpbatch-{name}", barrier=True)
# 
#         # start_time(start_inner_timer)
#         # col_mask = torch.cuda.BoolTensor(batch_mtx._nnz()).fill_(0)
#         # rowselect_csr_gpu(next_frontier_unique, batch_mtx.crow_indices(), col_mask, \
#         #                         next_frontier_unique.size(0), batch_mtx._nnz())
# 
#         # sample_row_lengths = batch_mtx.crow_indices()[1:] - batch_mtx.crow_indices()[:-1]
#         # sample_row_lengths = sample_row_lengths[next_frontier_unique]
#         # sample_crow_indices = torch.cuda.LongTensor(next_frontier_unique.size(0) + 1)
#         # sample_crow_indices[0] = 0
#         # sample_crow_indices[1:] = torch.cumsum(sample_row_lengths, dim=0)
#         # stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-cols-{name}", barrier=True)
#         # 
#         # start_time(start_inner_timer)
#         # sample_cols = batch_mtx.col_indices()[col_mask] 
#         # sample_vals = batch_mtx.values()[col_mask] 
# 
#         # sample_mtxs[i] = torch.sparse_csr_tensor(sample_crow_indices, sample_cols, sample_vals, \
#         #                                     size=torch.Size([next_frontier_unique.size(0), batch_size]))
#         # sample_mtxs[i] = sample_mtxs[i].t().to_sparse_coo()
#         # sample_mtxs[i] = torch.sparse_coo_tensor(sample_mtxs[i]._indices(), sample_mtxs[i]._values(),
#         #                                             size=torch.Size([batch_size, next_frontier_select.size(1)]))
#         # stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-extract-sample-{name}", barrier=True)
# 
#     stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-extract-iters-{name}", barrier=True)
# 
#     start_time(start_timer)
#     sample_mtx = torch.cat(sample_mtxs, dim=0)
#     row_start = (mb_count // replication) * batch_size * rank_col
#     sample_mtx_rows = sample_mtx._indices()[0,:] + row_start
#     sample_mtx_indices = torch.stack((sample_mtx_rows, sample_mtx._indices()[1,:]))
#     sample_mtx = torch.sparse_coo_tensor(sample_mtx_indices, sample_mtx._values(), \
#                         size=torch.Size([mb_count * batch_size, next_frontier_select.size(1)])).to_sparse_csr()
# 
# 
#     rank_row_start = rank_c * replication
#     rank_row_stop = (rank_c + 1) * replication - 1
#     sample_mtx = sample_mtx.to_sparse_csr()
#     sample_mtx = csr_allreduce(sample_mtx, rank_row_start, rank_row_stop, rank)
#     sample_mtx = sample_mtx.to_sparse_coo()
#     stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-extract-reduce-{name}", barrier=True)
# 
#     return sample_mtx._indices(), sample_mtx._values()

def dist_spgemm15D(mata, matb, replication, rank, size, row_groups, col_groups, name, timing_dict=None):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_time(start_timer)
    chunk_size = math.ceil(float(mata.size(1) / (size / replication)))
    matc = torch.sparse_coo_tensor(size=(mata.size(0), matb.size(1))).cuda()
    rank_c = rank // replication
    rank_col = rank % replication
    stages = size // (replication ** 2)
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-inst-{name}", barrier=True)

    for i in range(stages):
        start_time(start_timer)
        q = (rank_col * (size // (replication ** 2)) + i) * replication + rank_col

        matb_recv_nnz = torch.cuda.IntTensor([matb._nnz()])
        dist.broadcast(matb_recv_nnz, src=q, group=col_groups[rank_col])
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-bcast-nnz-{name}", barrier=True)

        start_time(start_timer)
        if q == rank:
            matb_recv_indices = matb._indices().clone()
            matb_recv_values = matb._values().clone()
        else:
            matb_recv_indices = torch.cuda.LongTensor(2, matb_recv_nnz.item())
            matb_recv_values = torch.cuda.DoubleTensor(matb_recv_nnz.item())
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-inst-recv-{name}", barrier=True)

        start_time(start_timer)
        dist.broadcast(matb_recv_indices, src=q, group=col_groups[rank_col])
        dist.broadcast(matb_recv_values, src=q, group=col_groups[rank_col])
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-bcast-data-{name}", barrier=True)

        start_time(start_timer)
        am_partid = rank_col * (size // replication ** 2) + i
        chunk_col_start = am_partid * chunk_size
        chunk_col_stop = min((am_partid + 1) * chunk_size, mata.size(1))
        chunk_col_size = chunk_col_stop - chunk_col_start
        chunk_col_mask = (mata._indices()[1, :] >= chunk_col_start) & (mata._indices()[1, :] < chunk_col_stop)

        mata_chunk_indices = mata._indices()[:, chunk_col_mask]
        mata_chunk_indices[1,:] -= chunk_col_start
        mata_chunk_values = mata._values()[chunk_col_mask]
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-local-{name}", barrier=True)

        start_time(start_timer)
        matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_chunk_indices, \
                                                    mata_chunk_values.double(), matb_recv_indices, \
                                                    matb_recv_values, mata.size(0), \
                                                    chunk_col_size, matb.size(1), coalesced=True)
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-local-spgemm-{name}", barrier=True)

        start_time(start_timer)
        # matc_chunk = torch.sparse_coo_tensor(matc_chunk_indices, matc_chunk_values, size=matc.size())
        matc_chunk = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, matc.size())
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-chunk-inst-{name}", barrier=True)

        start_time(start_timer)
        matc_chunk = matc_chunk.coalesce()
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-chunk-coalesce-{name}", barrier=True)

        start_time(start_timer)
        matc += matc_chunk
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-chunk-add-{name}", barrier=True)

    start_time(start_timer)
    matc = matc.coalesce()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-coalesce-{name}", barrier=True)

    # dist.all_reduce(matc, op=dist.reduce_op.SUM, group=row_groups[rank_c])
    # Implement sparse allreduce w/ all_gather and padding
    start_time(start_timer)
    matc_nnz = torch.cuda.IntTensor(1).fill_(matc._nnz())
    dist.all_reduce(matc_nnz, dist.ReduceOp.MAX, row_groups[rank_c])
    matc_nnz = matc_nnz.item()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-nnz-{name}", barrier=True)

    start_time(start_timer)
    matc_recv_indices = []
    matc_recv_values = []
    for i in range(replication):
        matc_recv_indices.append(torch.cuda.LongTensor(2, matc_nnz).fill_(0))
        matc_recv_values.append(torch.cuda.DoubleTensor(matc_nnz).fill_(0.0))

    matc_send_indices = torch.cat((matc._indices(), torch.cuda.LongTensor(2, matc_nnz - matc._nnz()).fill_(0)), 1)
    matc_send_values = torch.cat((matc._values(), torch.cuda.DoubleTensor(matc_nnz - matc._nnz()).fill_(0.0)))
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-padding-{name}", barrier=True)

    start_time(start_timer)
    dist.all_gather(matc_recv_indices, matc_send_indices, row_groups[rank_c])
    dist.all_gather(matc_recv_values, matc_send_values, row_groups[rank_c])
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-allgather-{name}", barrier=True)

    start_time(start_timer)
    matc_recv = []
    for i in range(replication):
        # matc_recv.append(torch.sparse_coo_tensor(matc_recv_indices[i], matc_recv_values[i], matc.size()))
        matc_recv.append(sparse_coo_tensor_gpu(matc_recv_indices[i], matc_recv_values[i], matc.size()))
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-reduce-{name}", barrier=True)

    start_time(start_timer)
    matc_recv = torch.stack(matc_recv)
    matc = torch.sparse.sum(matc_recv, dim=0)
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-{name}", barrier=True)
    start_time(start_timer)
    matc = matc.coalesce()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-coalesce-{name}", barrier=True)

    start_time(start_timer)
    nnz_mask = matc._values() > 0
    matc_nnz_indices = matc._indices()[:, nnz_mask]
    matc_nnz_values = matc._values()[nnz_mask]
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-unpad-{name}", barrier=True)

    return matc_nnz_indices, matc_nnz_values

def dist_saspgemm15D(mata, matb, replication, rank, size, row_groups, col_groups, \
                            name, nnz_row_masks, timing_dict=None, alg=None):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_inner_timer = torch.cuda.Event(enable_timing=True)
    stop_inner_timer = torch.cuda.Event(enable_timing=True)

    start_time(start_timer)
    # chunk_size = math.ceil(float(mata.size(1) / (size / replication)))
    chunk_size = mata.size(1) // (size // replication)
    if True or mata.layout == torch.sparse_coo or (name == "prob" and alg == "sage"):
        matc = torch.sparse_coo_tensor(size=(mata.size(0), matb.size(1))).cuda()
    elif mata.layout == torch.sparse_csr:
        matc_crow_indices = torch.cuda.LongTensor(mata.size(0) + 1).fill_(0)
        matc = torch.sparse_csr_tensor(matc_crow_indices, \
                                            torch.cuda.LongTensor(0), \
                                            torch.cuda.FloatTensor(0),
                                            size=(mata.size(0), matb.size(1)))
    rank_c = rank // replication
    rank_col = rank % replication
    stages = size // (replication ** 2)
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-inst-{name}", barrier=True)

    for i in range(stages):
        # start_time(start_timer)
        q = (rank_col * (size // (replication ** 2)) + i) * replication + rank_col

        # Extract chunk of mata for local SpGEMM
        start_time(start_timer)
        am_partid = rank_col * (size // replication ** 2) + i
        chunk_col_start = am_partid * chunk_size
        chunk_col_stop = min((am_partid + 1) * chunk_size, mata.size(1))
        chunk_col_size = chunk_col_stop - chunk_col_start
        if mata.layout == torch.sparse_coo:
            chunk_col_mask = (mata._indices()[1, :] >= chunk_col_start) & \
                             (mata._indices()[1, :] < chunk_col_stop)
            mata_chunk_indices = mata._indices()[:, chunk_col_mask]
            mata_chunk_indices[1,:] -= chunk_col_start
            mata_chunk_values = mata._values()[chunk_col_mask]
        elif mata.layout == torch.sparse_csr:
            # Column selection
            chunk_col_mask = (mata.col_indices() >= chunk_col_start) & \
                             (mata.col_indices() < chunk_col_stop)
            mata_chunk_cols = mata.col_indices()[chunk_col_mask]
            mata_chunk_cols -= chunk_col_start
            mata_chunk_values = mata.values()[chunk_col_mask]

            # # torch 1.13 supports offsets-based reduction
            # mata_chunk_rowcount = torch.segment_reduce(chunk_col_mask.float(), \
            #                                                 "sum", \
            #                                                 offsets=mata.crow_indices())

            lengths = mata.crow_indices()[1:] - mata.crow_indices()[:-1]
            mata_chunk_rowcount = torch.segment_reduce(chunk_col_mask.float(), \
                                                            "sum", \
                                                            lengths=lengths)
            mata_chunk_crows = torch.cuda.LongTensor(mata.size(0) + 1)
            mata_chunk_crows[1:] = torch.cumsum(mata_chunk_rowcount.long(), dim=0)
            mata_chunk_crows[0] = 0

        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-local-{name}", barrier=True)

        # Determine number of nonzero columns in chunk of mata
        start_time(start_timer)
        if mata.layout == torch.sparse_coo:
            nnz_cols = torch.unique(mata_chunk_indices[1,:])
        elif mata.layout == torch.sparse_csr:
            nnz_cols = torch.unique(mata_chunk_cols)
        nnz_cols_count = torch.cuda.IntTensor(1).fill_(nnz_cols.size(0))
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-unique-{name}", barrier=True)
        # timing_dict[f"spgemm-nnzcount-{name}"].append(nnz_cols.size(0))

        # Gather nnz column counts on rank q
        start_time(start_timer)
        # print(f"nnz_cols_count: {nnz_cols_count} q: {q} rank_col: {rank_col}", flush=True)
        torch.cuda.synchronize()
        if rank == q:
            nnz_cols_count_list = []
            for j in range(size // replication):
                nnz_cols_count_list.append(torch.cuda.IntTensor(1).fill_(0))
                recv_rank = rank_col + j * replication
            dist.gather(nnz_cols_count, nnz_cols_count_list, dst=q, group=col_groups[rank_col])
        else:
            dist.gather(nnz_cols_count, dst=q, group=col_groups[rank_col])
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-gather-nnzcounts-{name}", barrier=True)

        # Rank q allocates recv buffer for nnz col ids
        start_time(start_timer)
        if rank == q:
            nnz_col_ids = []
            for nnz_count in nnz_cols_count_list:
                nnz_col_ids.append(torch.cuda.LongTensor(nnz_count.item()).fill_(0))
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-alloc-nnzbuff-{name}", barrier=True)
        
        # isend/irecv nnz col ids to rank q
        start_time(start_timer)
        if rank == q:
            recv_objs = []
            for j in range(size // replication):
                recv_rank = rank_col + j * replication
                if recv_rank != q:
                    recv_objs.append(dist.recv(nnz_col_ids[j], src=recv_rank))
                else:
                    nnz_col_ids[j] = nnz_cols
        else:
            dist.send(nnz_cols, dst=q)
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-send-colids-{name}", barrier=True)

        # Send selected rows count 
        start_time(start_timer)
        nnz_count = 0
        if rank == q:
            nnz_cols_count_list = torch.cat(nnz_cols_count_list, dim=0)
            if matb.layout == torch.sparse_coo:
                start_time(start_inner_timer)
                # print(f"nnz_col_ids.size: {nnz_col_ids.size()}", flush=True)
                # print(f"nnz_col_ids: {nnz_col_ids}", flush=True)
                rowselect_coo_gpu(nnz_col_ids, matb._indices()[0,:].long(), nnz_row_masks, nnz_cols_count_list.int(), \
                                        matb._indices().size(1), size // replication)
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-rowsel-{name}")
            recv_proc_count = ((size // replication) - 1)
            send_ops = [None] * 4 * recv_proc_count
            send_idx = 0

            # start_time(start_inner_timer)
            for j in range(size // replication):
                recv_rank = rank_col + j * replication
                # nnz_row_mask = nnz_row_masks[(j * matb._nnz()):((j + 1) * matb._nnz())]
                nnz_row_mask = torch.cuda.BoolTensor(matb._nnz()).fill_(False)
                
                if matb.layout == torch.sparse_csr:
                    start_time(start_inner_timer)
                    rowselect_csr_gpu(nnz_col_ids[j], matb.crow_indices().long(), nnz_row_mask, \
                                            nnz_col_ids[j].size(0), matb._nnz())
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-rowsel-{name}")

                    start_time(start_inner_timer)
                    row_lengths = matb.crow_indices()[1:] - matb.crow_indices()[:-1]
                    if nnz_col_ids[j].size(0) > 0:
                        matb_send_lengths = row_lengths[nnz_col_ids[j]]
                    else:
                        matb_send_lengths = torch.cuda.LongTensor(0)

                    matb_send_cols = matb.col_indices()[nnz_row_mask]
                    matb_send_values = matb.values()[nnz_row_mask]
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-misc-{name}")

                    start_time(start_inner_timer)
                    if recv_rank != q:
                        selected_rows_count = torch.cuda.IntTensor(1).fill_(matb_send_cols.size(0))
                        nnz_count += 2 * matb_send_cols.size(0) + nnz_col_ids[j].size(0)
                        # dist.send(selected_rows_count, tag=0, dst=recv_rank)
                        # # dist.send(matb_send_crows, tag=1, dst=recv_rank)
                        # dist.send(matb_send_lengths, tag=1, dst=recv_rank)
                        # dist.send(matb_send_cols, tag=2, dst=recv_rank)
                        # dist.send(matb_send_values, tag=3, dst=recv_rank)

                        # print(f"send selected_rows_count: {selected_rows_count}", flush=True)
                        send_ops[send_idx] = \
                                    dist.P2POp(dist.isend, selected_rows_count, recv_rank, tag=0)
                        send_ops[send_idx + 1] = \
                                    dist.P2POp(dist.isend, matb_send_lengths.int(), recv_rank, tag=1)
                        send_ops[send_idx + 2] = \
                                    dist.P2POp(dist.isend, matb_send_cols.int(), recv_rank, tag=2)
                        send_ops[send_idx + 3] = \
                                    dist.P2POp(dist.isend, matb_send_values.float(), recv_rank, tag=3)
                        send_idx += 4
                    else:
                        # matb_recv_crows = matb_send_crows.clone()
                        matb_recv_cols = matb_send_cols.clone()
                        matb_recv_values = matb_send_values.clone()
                        if not (name == "prob" and alg == "sage"):
                            # matb_rows_sum = torch.cuda.LongTensor(matb.crow_indices().size(0) - 1).fill_(0)
                            matb_rows_sum = torch.cuda.IntTensor(matb.crow_indices().size(0) - 1).fill_(0)
                            if nnz_row_mask.any():
                                matb_rows_sum[nnz_col_ids[j]] = matb_send_lengths
                            # matb_recv_crows = torch.cuda.LongTensor(matb.crow_indices().size(0))
                            # matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dim=0)
                            matb_recv_crows = torch.cuda.IntTensor(matb.crow_indices().size(0))
                            matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dtype=torch.int32, dim=0)
                            matb_recv_crows[0] = 0
                        else:
                            # matb_recv_rows = torch.repeat_interleave(nnz_col_ids[j], matb_send_lengths)
                            # matb_recv_indices = torch.stack((matb_recv_rows, matb_recv_cols))
                            # matb_rows_sum = torch.cuda.LongTensor(matb.crow_indices().size(0) - 1).fill_(0)
                            matb_rows_sum = torch.cuda.IntTensor(matb.crow_indices().size(0) - 1).fill_(0)
                            if nnz_row_mask.any():
                                matb_rows_sum[nnz_col_ids[j]] = matb_send_lengths
                            # matb_recv_crows = torch.cuda.LongTensor(matb.crow_indices().size(0))
                            matb_recv_crows = torch.cuda.IntTensor(matb.crow_indices().size(0))
                            matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dtype=torch.int32, dim=0)
                            matb_recv_crows[0] = 0
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-calls-{name}")
                elif matb.layout == torch.sparse_coo:
                    # rowselect_coo_gpu(nnz_col_ids[j], matb._indices()[0,:], nnz_row_mask, \
                    #                         nnz_col_ids[j].size(0), matb._indices().size(1))

                    start_time(start_inner_timer)
                    matb_send_indices = matb._indices()[:, nnz_row_mask]
                    matb_send_values = matb._values()[nnz_row_mask]
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-misc-{name}")

                    start_time(start_inner_timer)
                    if recv_rank != q:
                        selected_rows_count = torch.cuda.IntTensor(1).fill_(matb_send_indices.size(1))
                        nnz_count += 3 * matb_send_indices.size(1)
                        dist.send(selected_rows_count, tag=0, dst=recv_rank)
                        dist.send(matb_send_indices, tag=1, dst=recv_rank)
                        dist.send(matb_send_values, tag=2, dst=recv_rank)

                        # send_ops[send_idx] = dist.P2POp(dist.isend, selected_rows_count, recv_rank, tag=0)
                        # send_ops[send_idx + 1] = dist.P2POp(dist.isend, matb_send_indices, recv_rank, tag=1)
                        # send_ops[send_idx + 2] = dist.P2POp(dist.isend, matb_send_values, recv_rank, tag=2)
                        # send_idx += 3
                    else:
                        # matb_select_recv = matb_send.clone()
                        matb_recv_indices = matb_send_indices.clone()
                        matb_recv_values = matb_send_values.clone()
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-calls-{name}")

            # if len(send_ops) > 0:
            if len(send_ops) > 0 and matb.layout == torch.sparse_csr:
                reqs = dist.batch_isend_irecv(send_ops)
                for req in reqs:
                    req.wait()
        else:
            if f"spgemm-send-rowsel-{name}" not in timing_dict:
                timing_dict[f"spgemm-send-rowsel-{name}"] = []
            
            if f"spgemm-send-misc-{name}" not in timing_dict:
                timing_dict[f"spgemm-send-misc-{name}"] = []

            if f"spgemm-send-calls-{name}" not in timing_dict:
                timing_dict[f"spgemm-send-calls-{name}"] = []

            start_time(start_inner_timer)
            selected_rows_count_recv = torch.cuda.IntTensor(1).fill_(-1)
            recv_ops = [dist.P2POp(dist.irecv, selected_rows_count_recv, q, tag=0)]
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()
            torch.cuda.synchronize()
            # print(f"recv selected_rows_count: {selected_rows_count_recv}", flush=True)
            # dist.recv(selected_rows_count_recv, tag=0, src=q)

            if matb.layout == torch.sparse_csr:
                # matb_recv_crows = torch.cuda.LongTensor(chunk_col_size + 1)
                # dist.recv(matb_recv_crows, tag=1, src=q)

                # matb_recv_lengths = torch.cuda.LongTensor(nnz_cols.size(0))
                recv_ops = []
                matb_recv_lengths = torch.cuda.IntTensor(nnz_cols.size(0))
                # dist.recv(matb_recv_lengths, tag=1, src=q)
                # print(f"recv matb_recv_lengths: {matb_recv_lengths}", flush=True)
                recv_ops.append(dist.P2POp(dist.irecv, matb_recv_lengths, q, tag=1))
                # matb_recv_lengths = matb_recv_lengths.long()

                # matb_recv_cols = torch.cuda.LongTensor(selected_rows_count_recv.item())
                matb_recv_cols = torch.cuda.IntTensor(selected_rows_count_recv.item())
                # dist.recv(matb_recv_cols, tag=2, src=q)
                recv_ops.append(dist.P2POp(dist.irecv, matb_recv_cols, q, tag=2))
                # matb_recv_cols = matb_recv_cols.long()

                # matb_recv_values = torch.cuda.DoubleTensor(selected_rows_count_recv.item())
                matb_recv_values = torch.cuda.FloatTensor(selected_rows_count_recv.item())
                # dist.recv(matb_recv_values, tag=3, src=q)
                recv_ops.append(dist.P2POp(dist.irecv, matb_recv_values, q, tag=3))

                reqs = dist.batch_isend_irecv(recv_ops)
                for req in reqs:
                    req.wait()
                torch.cuda.synchronize()

                # the spgemm is better with COO for SAGE prob
                if not (name == "prob" and alg == "sage"):
                    matb_recv_lengths = matb_recv_lengths.long()
                    matb_recv_cols = matb_recv_cols.long()
                    matb_rows_sum = torch.cuda.LongTensor(chunk_col_size).fill_(0)
                    if matb_recv_lengths.size(0) > 0:
                        matb_rows_sum[nnz_cols] = matb_recv_lengths
                    matb_recv_crows = torch.cuda.LongTensor(chunk_col_size + 1)
                    matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dtype=torch.int32, dim=0)
                    matb_recv_crows[0] = 0
                else:
                    # matb_recv_rows = torch.repeat_interleave(nnz_cols, matb_recv_lengths)
                    # matb_recv_indices = torch.stack((matb_recv_rows, matb_recv_cols))
                    # matb_rows_sum = torch.cuda.LongTensor(chunk_col_size).fill_(0)
                    matb_rows_sum = torch.cuda.IntTensor(chunk_col_size).fill_(0)
                    if matb_recv_lengths.size(0) > 0:
                        matb_rows_sum[nnz_cols] = matb_recv_lengths
                    # matb_recv_crows = torch.cuda.LongTensor(chunk_col_size + 1)
                    matb_recv_crows = torch.cuda.IntTensor(chunk_col_size + 1)
                    # matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dim=0)
                    matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dtype=torch.int32, dim=0)
                    matb_recv_crows[0] = 0

                nnz_count += 2 * matb_recv_cols.size(0) + matb_recv_lengths.size(0)

            elif matb.layout == torch.sparse_coo:

                # matb_indices_recv = matb_recv_buff[:(3 * selected_rows_count_recv.item())].view(3, -1)
                matb_recv_indices = torch.cuda.LongTensor(2, selected_rows_count_recv.item())
                dist.recv(matb_recv_indices, tag=1, src=q)

                matb_recv_values = torch.cuda.DoubleTensor(selected_rows_count_recv.item())
                dist.recv(matb_recv_values, tag=2, src=q)

                # recv_ops = [dist.P2POp(dist.irecv, matb_recv_indices, q, tag=1), \
                #                 dist.P2POp(dist.irecv, matb_recv_values, q, tag=2)]

                # reqs = dist.batch_isend_irecv(recv_ops)
                # for req in reqs:
                #     req.wait()
                nnz_count += 3 * matb_recv_indices.size(1)
            stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-recv-calls-{name}")

        torch.cuda.synchronize()
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-send-rowdata-{name}", barrier=True)

        if f"spgemm-recv-calls-{name}" not in timing_dict:
            timing_dict[f"spgemm-recv-calls-{name}"] = []

        timing_dict[f"spgemm-send-rownnz-{name}"].append(nnz_count)

        # start_time(start_timer)
        # # print(f"matb_select_recv.dtype: {matb_select_recv.dtype}")
        # # matb_recv_indices = matb_select_recv[:2, :].long()
        # # # matb_recv_values = matb_select_recv[2, :].double()
        # # matb_recv_values = matb_select_recv[2, :]

        # # matb_recv_indices = matb_select_recv[:2, :].long()
        # # matb_recv_values = matb_select_recv[2, :].double()
        # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-castrecv-{name}", barrier=True)

        start_time(start_timer)
        if mata_chunk_values.size(0) > 0 and matb_recv_values.size(0) > 0:
            # Can skip local spgemm for row-selection for any alg and for SAGE's probability spgemm
            # if name == "prob" and alg == "sage":
            #     matc_chunk_crows = torch.cuda.LongTensor(mata.size(0)).fill_(0)
            #     matc_chunk = torch.sparse_csr_tensor(matb_recv_crows, matb_recv_cols, matb_recv_values.float(),
            #                                             size=(chunk_col_size, matb.size(1)))
            #     matc += matc_chunk
            if mata.layout == torch.sparse_csr and (name == "prob" and alg == "sage") or (name == "rowsel"):
                start_time(start_inner_timer)
                # mata_recv = torch.sparse_csr_tensor(mata_chunk_crows, mata_chunk_cols, mata_chunk_values,
                #                                         size=(mata.size(0), chunk_col_size))
                mata_recv_row_lens = mata_chunk_crows[1:] - mata_chunk_crows[:-1]
                mata_recv_rows = torch.repeat_interleave(
                                    torch.arange(0, mata.size(0), device=mata.device), 
                                    mata_recv_row_lens)
                # matb_recv = sparse_coo_tensor_gpu(matb_recv_indices, matb_recv_values.float(),
                #                                         torch.Size([chunk_col_size, matb.size(1)]))
                # matb_recv = matb_recv.to_sparse_csr()
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-csrinst-{name}")



                start_time(start_inner_timer)
                # mata_recv = mata_recv.to_sparse_coo()
                # matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_recv._indices(), \
                #                                             mata_recv._values().float(), matb_recv_indices, \
                #                                             matb_recv_values.float(), mata.size(0), \
                #                                             chunk_col_size, matb.size(1), coalesced=True)
                # print(f"mata_nnz: {mata_chunk_values.size()} matb_nnz: {matb_recv._nnz()} dims: {mata.size(0)} {chunk_col_size} {matb_recv.size(1)}", flush=True)

                # matb_recv_row_lens = matb_recv.crow_indices()[1:] - matb_recv.crow_indices()[:-1]
                matb_recv_row_lens = matb_recv_crows[1:] - matb_recv_crows[:-1]
                # matc_chunk_row_lens = torch.cuda.LongTensor(mata.size(0) + 1).fill_(0)
                matc_chunk_row_lens = torch.cuda.IntTensor(mata.size(0) + 1).fill_(0)
                matc_chunk_row_lens[mata_recv_rows] = matb_recv_row_lens[mata_chunk_cols].int()
                matc_chunk_crows = torch.cumsum(matc_chunk_row_lens, 0).roll(1)
                matc_chunk_crows[0] = 0

                # matc_chunk_cols = torch.cuda.LongTensor(matc_chunk_crows[-1].item()).fill_(0)
                matc_chunk_cols = torch.cuda.IntTensor(matc_chunk_crows[-1].item()).fill_(0)
                rearrange_rows_gpu(mata_recv_rows, mata_chunk_cols, matc_chunk_crows, 
                # rearrangel_rows_gpu(mata_recv_rows, mata_chunk_cols, matc_chunk_crows, 
                                        matb_recv_crows.int(), matb_recv_cols.int(), matc_chunk_cols)
                                        # matb_recv.crow_indices(), matb_recv.col_indices(), matc_chunk_cols)
                matc_chunk_values = torch.cuda.FloatTensor(matc_chunk_crows[-1].item()).fill_(1.0)

                # matc_chunk = nsparse_spgemm(mata_chunk_crows.int(), \
                #                         mata_chunk_cols.int(), \
                #                         mata_chunk_values.float(), \
                #                         matb_recv.crow_indices().int(), \
                #                         matb_recv.col_indices().int(), \
                #                         matb_recv.values().float(), \
                #                         mata.size(0), chunk_col_size, matb_recv.size(1))
                # matc_chunk_crows = matc_chunk[0]
                # matc_chunk_cols = matc_chunk[1]
                # matc_chunk_values = matc_chunk[2]

                # mata_recv = mata_recv.to_sparse_coo()
                # matb_recv = matb_recv.to_sparse_coo()
                # matc_chunk = torch.sparse.mm(mata_recv.float(), matb_recv.float())
                # matc_chunk_crows = matc_chunk.crow_indices()
                # matc_chunk_cols = matc_chunk.col_indices()
                # matc_chunk_values = matc_chunk.values()
                matc_row_lens = matc_chunk_crows[1:] - matc_chunk_crows[:-1]
                matc_chunk_rows = torch.repeat_interleave(
                                    torch.arange(0, mata.size(0), device=mata.device), 
                                    matc_row_lens)
                matc_chunk_indices = torch.stack((matc_chunk_rows, matc_chunk_cols))

                matc_chunk = torch.sparse_coo_tensor(matc_chunk_indices, matc_chunk_values, size=matc.size())
                matc_chunk = matc_chunk.coalesce()
                matc += matc_chunk
                torch.cuda.synchronize()
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-spspmm-{name}")
                # matc = matc.to_sparse_csr()
            elif mata.layout == torch.sparse_csr:
                start_time(start_inner_timer)
                mata_recv = torch.sparse_csr_tensor(mata_chunk_crows, mata_chunk_cols, mata_chunk_values.float(),
                                                        size=(mata.size(0), chunk_col_size))
                matb_recv = torch.sparse_csr_tensor(matb_recv_crows, matb_recv_cols, matb_recv_values.float(),
                                                        size=(chunk_col_size, matb.size(1)))
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-csrinst-{name}")


                start_time(start_inner_timer)
                mata_recv = mata_recv.to_sparse_coo()
                matb_recv = matb_recv.to_sparse_coo()
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-csr2coo-{name}")

                # matc += torch.mm(mata_recv, matb_recv)
                start_time(start_inner_timer)
                matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_recv._indices(), \
                                                            mata_recv._values(), matb_recv._indices(), \
                                                            matb_recv._values(), mata.size(0), \
                                                            chunk_col_size, matb.size(1), coalesced=True)

                # mata_row_lens = mata_chunk_crows[1:] - mata_chunk_crows[:-1]
                # torch.cuda.synchronize()
                # print(f"before nsparse", flush=True)
                # matc_chunk = nsparse_spgemm(mata_chunk_crows.int(), \
                #                         mata_chunk_cols.int(), \
                #                         mata_chunk_values.float(), \
                #                         matb_recv_crows.int(), \
                #                         matb_recv_cols.int(), \
                #                         matb_recv_values.float(), \
                #                         mata.size(0), chunk_col_size, matb.size(1))
                # print(f"after nsparse", flush=True)
                # matc_chunk_crows = matc_chunk[0]
                # matc_chunk_cols = matc_chunk[1]
                # matc_chunk_values = matc_chunk[2]

                # matc_row_lens = matc_chunk_crows[1:] - matc_chunk_crows[:-1]
                # matc_chunk_rows = torch.repeat_interleave(
                #                     torch.arange(0, mata.size(0), device=mata.device), 
                #                     matc_row_lens)
                # matc_chunk_indices = torch.stack((matc_chunk_rows, matc_chunk_cols))

                matc_chunk = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, 
                                                        torch.Size([matc.size(0), matc.size(1)]))
                matc_chunk = matc_chunk.coalesce()

                matc += matc_chunk
                matc_chunk = matc_chunk.to_sparse_csr()
                # matc += torch.mm(mata_recv, matb_recv)
                # matc_offsets, matc_cols, matc_values = spgemm_gpu(mata_recv.crow_indices().int(), mata_recv.col_indices().int(), mata_recv.values(), matb_recv.crow_indices().int(), matb_recv.col_indices().int(), matb_recv.values(), matc.crow_indices().int(), matc.col_indices().int(), matc.values(), int(mata_recv.size(0)), int(mata_recv.size(1)), int(matb_recv.size(1)))
                # matc = torch.sparse_csr_tensor(matc_offsets, matc_cols, matc_values, size=matc.size())
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-spspmm-{name}")
                # matc = matc.to_sparse_csr()
            elif mata.layout == torch.sparse_coo:
                matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_chunk_indices, \
                                                            mata_chunk_values, matb_recv_indices, \
                                                            matb_recv_values, mata.size(0), \
                                                            chunk_col_size, matb.size(1), coalesced=True)
                matc_chunk = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, 
                                                        torch.Size([matc.size(0), matc.size(1)]))
                # matc_chunk = matc_chunk.coalesce()
                matc += matc_chunk
                if f"spgemm-loc-csrinst-{name}" not in timing_dict:
                    timing_dict[f"spgemm-loc-csrinst-{name}"] = []

                if f"spgemm-loc-csr2coo-{name}" not in timing_dict:
                    timing_dict[f"spgemm-loc-csr2coo-{name}"] = []

                if f"spgemm-loc-spspmm-{name}" not in timing_dict:
                    timing_dict[f"spgemm-loc-spspmm-{name}"] = []
        else:
            if f"spgemm-loc-csrinst-{name}" not in timing_dict:
                timing_dict[f"spgemm-loc-csrinst-{name}"] = []

            # if f"spgemm-loc-csr2coo-{name}" not in timing_dict:
            #     timing_dict[f"spgemm-loc-csr2coo-{name}"] = []

            if f"spgemm-loc-spspmm-{name}" not in timing_dict:
                timing_dict[f"spgemm-loc-spspmm-{name}"] = []

        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-local-spgemm-{name}", barrier=True)

    # print(f"send-rowdata: {timing_dict['spgemm-send-rowdata-prob']}")
    # print(f"send-rownnz: {timing_dict['spgemm-send-rownnz-prob']}")
    # print(f"nnzcount: {timing_dict['spgemm-nnzcount-prob']}")

    rank_row_start = rank_c * replication
    rank_row_stop = (rank_c + 1) * replication - 1

    start_time(start_timer)
    matc = matc.to_sparse_csr()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-coo2csr{name}", barrier=True)

    start_time(start_timer)
    # matc = csr_allreduce(matc, rank_row_start, rank_row_stop, rank)
    matc = csr_allreduce(matc, rank_row_start, rank_row_stop, rank, "prob", alg, timing_dict)
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-{name}", barrier=True)

    start_time(start_timer)
    matc = matc.to_sparse_coo()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-csr2coo{name}", barrier=True)
    return matc._indices(), matc._values()

    # # Old sparse reduction
    # start_time(start_timer)
    # if matc.layout == torch.sparse_csr:
    #     matc = matc.to_sparse_coo()
    # matc = matc.coalesce()
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-coalesce-{name}", barrier=True)

    # # Implement sparse allreduce w/ all_gather and padding
    # start_time(start_timer)
    # matc_nnz = torch.cuda.IntTensor(1).fill_(matc._nnz())
    # dist.all_reduce(matc_nnz, dist.ReduceOp.MAX, row_groups[rank_c])
    # matc_nnz = matc_nnz.item()
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-nnz-{name}", barrier=True)

    # torch.cuda.nvtx.range_push("nvtx-padding")
    # start_time(start_timer)
    # matc_recv_indices = []
    # matc_recv_values = []
    # for i in range(replication):
    #     matc_recv_indices.append(torch.cuda.LongTensor(2, matc_nnz).fill_(0))
    #     matc_recv_values.append(torch.cuda.DoubleTensor(matc_nnz).fill_(0.0))

    # matc_send_indices = torch.cat((torch.cuda.LongTensor(2, matc_nnz - matc._nnz()).fill_(0), matc._indices()), 1)
    # matc_send_values = torch.cat((torch.cuda.DoubleTensor(matc_nnz - matc._nnz()).fill_(0.0), matc._values()))
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-padding-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # padding

    # torch.cuda.nvtx.range_push("nvtx-allgather")
    # start_time(start_timer)
    # dist.all_gather(matc_recv_indices, matc_send_indices, row_groups[rank_c])
    # dist.all_gather(matc_recv_values, matc_send_values, row_groups[rank_c])
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-allgather-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # all-gather

    # torch.cuda.nvtx.range_push("nvtx-preproc-reduce")
    # start_time(start_timer)
    # matc_recv = []
    # for i in range(replication):
    #     # matc_recv.append(torch.sparse_coo_tensor(matc_recv_indices[i], matc_recv_values[i], matc.size()))

    #     # Unclear why this is necessary but it seems to hang otherwise
    #     nnz_mask = matc_recv_values[i] > 0
    #     matc_nnz_indices = matc_recv_indices[i][:, nnz_mask]
    #     matc_nnz_values = matc_recv_values[i][nnz_mask]

    #     matc_recv.append(torch.sparse_coo_tensor(matc_nnz_indices, matc_nnz_values, matc.size()))
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-reduce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # preproc-reduce

    # torch.cuda.nvtx.range_push("nvtx-reduce")
    # start_time(start_timer)
    # # matc_recv = torch.stack(matc_recv)
    # # matc = torch.sparse.sum(matc_recv, dim=0)

    # gpu = torch.device(f"cuda:{torch.cuda.current_device()}")
    # for i in range(replication):
    #     recv_rank = rank_c * replication + i
    #     if recv_rank != rank:
    #         # matc += matc_recv[i]
    #         matc_indices = matc._indices().int()
    #         matc_recv_indices = matc_recv[i]._indices().int()
    #         matc_values = matc._values().double()

    #         matc_outputs = coogeam_gpu(matc_indices[0,:], matc_indices[1,:], matc_values, 
    #                             matc_recv_indices[0,:], matc_recv_indices[1,:], matc_recv[i]._values(),
    #                             matc.size(0), matc.size(1))
    #         matc_chunk_rows = matc_outputs[0].long()
    #         matc_chunk_cols = matc_outputs[1].long()
    #         matc_chunk_values = matc_outputs[2].double()
    #         matc_chunk_counts = torch.diff(matc_chunk_rows)
    #         matc_chunk_rows = torch.repeat_interleave(
    #                                     torch.arange(0, matc.size(0), device=gpu),
    #                                     matc_chunk_counts)
    #         matc_chunk_indices = torch.stack((matc_chunk_rows, matc_chunk_cols))
    #         matc = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, matc.size())
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # reduce

    # torch.cuda.nvtx.range_push("nvtx-reduce-coalesce")
    # start_time(start_timer)
    # matc = matc.coalesce()
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-coalesce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # reduce-coalesce

    # torch.cuda.nvtx.range_push("nvtx-unpad")
    # start_time(start_timer)
    # # nnz_mask = matc._values() != -1.0
    # nnz_mask = matc._values() > 0
    # matc_nnz_indices = matc._indices()[:, nnz_mask]
    # matc_nnz_values = matc._values()[nnz_mask]
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-unpad-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # unpad
    # 
    # return matc_nnz_indices, matc_nnz_values

def gen_prob_dist(numerator, adj_matrix, mb_count, node_count_total, replication, 
                    rank, size, row_groups, col_groups, sa_masks, 
                    timing_dict, name, timing_arg, replicate_graph):

    global timing
    timing = timing_arg

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    # TODO: assume n_layers=1 for now
    # start_time(start_timer)
    start_timer.record()
    # p_num_indices, p_num_values = dist_saspgemm15D(
    #                                     numerator, adj_matrix, replication, rank, size, 
    #                                     row_groups, col_groups, "prob", sa_masks, 
    #                                     timing_dict, name)
    if name == "sage" and replicate_graph:
        matc_chunk_row_lens = torch.cuda.LongTensor(numerator.size(0)).fill_(0)
        mata_rows = numerator._indices()[0,:]
        mata_cols = numerator._indices()[1,:]
        adj_row_lens = adj_matrix.crow_indices()[1:] - adj_matrix.crow_indices()[:-1]
        # print(f"matc_chunk_row_lens.size: {matc_chunk_row_lens.size()}", flush=True)
        # print(f"adj_row_lens.size: {adj_row_lens.size()}", flush=True)
        # print(f"mata_rows.min: {mata_rows.min()} max: {mata_rows.max()}", flush=True)
        # print(f"mata_cols.min: {mata_cols.min()} max: {mata_cols.max()}", flush=True)
        matc_chunk_row_lens[mata_rows] = adj_row_lens[mata_cols]
        # matc_chunk_crows = torch.cuda.LongTensor(numerator.size(0) + 1).fill_(0)
        matc_chunk_crows = torch.cuda.IntTensor(numerator.size(0) + 1).fill_(0)
        matc_chunk_crows[1:] = torch.cumsum(matc_chunk_row_lens, 0)
        matc_chunk_crows[0] = 0
        matc_chunk_cols = torch.cuda.IntTensor(matc_chunk_crows[-1].item()).fill_(0)
        rearrange_rows_gpu(numerator._indices()[0,:].long(), 
                                numerator._indices()[1,:].long(), 
                                matc_chunk_crows.long(), 
                                adj_matrix.crow_indices().int(), 
                                adj_matrix.col_indices().int(), 
                                matc_chunk_cols)
        matc_chunk_cols = matc_chunk_cols.long()
        # p_num_rows = torch.repeat_interleave(
        #                                 torch.arange(matc_chunk_crows.size(0) - 1, 
        #                                                     dtype=torch.int32, 
        #                                                     device=numerator.device),
        #                                 matc_chunk_row_lens)
        # p_num_indices = torch.stack((p_num_rows, matc_chunk_cols))
        p_num_crows = matc_chunk_crows
        p_num_cols = matc_chunk_cols
        # p_num_values = torch.cuda.FloatTensor(p_num_indices.size(1)).fill_(1.0)
        # p_num_values = torch.cuda.DoubleTensor(p_num_indices.size(1)).fill_(1.0)
        p_num_values = torch.cuda.DoubleTensor(p_num_cols.size(0)).fill_(1.0)
    else:
        if sa_masks is not None:
            sa_masks.fill_(False)
        p_num_indices, p_num_values = dist_saspgemm15D(
                                            numerator, adj_matrix, replication, rank, size, 
                                            row_groups, col_groups, "prob", sa_masks, 
                                            timing_dict, name)
        p_num_values = p_num_values.double()

    # p_num_indices, p_num_values =  dist_spgemm15D(numerator, adj_matrix, replication, rank, size, row_groups, col_groups, "prob")
    stop_timer.record()
    torch.cuda.synchronize()
    time_taken = start_timer.elapsed_time(stop_timer)
    # timing_dict["probability-spgemm"].append(stop_time(start_timer, stop_timer))
    timing_dict["probability-spgemm"].append(time_taken)

    start_time(start_timer)
    if name == "ladies":
        # p_num_values = torch.square(p_num_values).double()
        p_den = torch.cuda.DoubleTensor(numerator.size(0)).fill_(0)
        p_num_values = p_num_values.double()
        p_den.scatter_add_(0, p_num_indices[0, :].long(), p_num_values)
        normalize_gpu(p_num_values, p_den, p_num_indices[0, :].long(), p_num_values.size(0))
        p_num_values = torch.nan_to_num(p_num_values)
        p = sparse_coo_tensor_gpu(p_num_indices.long(), p_num_values, torch.Size([numerator.size(0), node_count_total]))
    elif name == "sage":
        # p_num_values = torch.cuda.DoubleTensor(p_num_values.size(0)).fill_(1.0)
        # p_num_values = p_num_values.double()
        # p_den = torch.cuda.FloatTensor(numerator.size(0)).fill_(0)
        # p_den = torch.cuda.DoubleTensor(numerator.size(0)).fill_(0)
        p_num_crows = p_num_crows.long()
        normalize_csr_gpu(p_num_values, p_num_crows, p_num_crows.size(0) - 1)
        p_num_values = torch.nan_to_num(p_num_values)
        p = torch.sparse_csr_tensor(p_num_crows, p_num_cols.long(), p_num_values, torch.Size([numerator.size(0), node_count_total]))
    # p_den.scatter_add_(0, p_num_indices[0, :].long(), p_num_values)
    # p = torch.sparse_coo_tensor(indices=p_num_indices, 
    #                                 values=p_num_values, 
    #                                 size=(numerator.size(0), node_count_total))
    # p = sparse_coo_tensor_gpu(p_num_indices, p_num_values, torch.Size([numerator.size(0), node_count_total]))
    # normalize_gpu(p._values(), p_den, p._indices()[0, :], p._nnz())
    torch.cuda.synchronize()
    timing_dict["compute-p"].append(stop_time(start_timer, stop_timer))
    return p

def sample(p, frontier_size, mb_count, node_count_total, n_darts, replication, 
                rank, size, row_groups, col_groups, timing_dict, name):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    sample_start_timer = torch.cuda.Event(enable_timing=True)
    sample_stop_timer = torch.cuda.Event(enable_timing=True)

    select_start_timer = torch.cuda.Event(enable_timing=True)
    select_stop_timer = torch.cuda.Event(enable_timing=True)

    select_iter_start_timer = torch.cuda.Event(enable_timing=True)
    select_iter_stop_timer = torch.cuda.Event(enable_timing=True)

    rank_c = rank // replication
    rank_col = rank % replication

    # n_darts_col = n_darts // replication
    # if rank_col == replication - 1:
    #     n_darts_col = n_darts - (replication - 1) * n_darts_col

    # n_darts_col = n_darts
    # avg_degree = int(p._nnz() / p.size(0))
    # n_darts_col = int(avg_degree * frontier_size / avg_degree)
    n_darts_col = frontier_size

    p_row_lens = p.crow_indices()[1:] - p.crow_indices()[:-1]
    next_frontier_rows = torch.repeat_interleave(
                                    torch.arange(p.crow_indices().size(0) - 1, 
                                                        dtype=torch.int32, 
                                                        device=p.device),
                                    p_row_lens)
    next_frontier_indices = torch.stack((next_frontier_rows, p.col_indices()))
    next_frontier = torch.sparse_coo_tensor(indices=next_frontier_indices,
                                        values=torch.cuda.LongTensor(p._nnz()).fill_(0),
                                        size=(p.size(0), node_count_total))
    # next_frontier = sparse_coo_tensor_gpu(p._indices(), torch.cuda.LongTensor(p._nnz()).fill_(0), 
    #                                             torch.Size([p.size(0), node_count_total]))

    start_time(start_timer)

    frontier_nnz_sizes = torch.cuda.IntTensor(p.size(0)).fill_(0)
    ones = torch.cuda.IntTensor(next_frontier._indices()[0,:].size(0)).fill_(1)
    if name == "ladies":
        frontier_nnz_sizes.scatter_add_(0, next_frontier._indices()[0,:], ones)
        zero_idxs = p._indices()[0, (p._values() == 0)].unique()
    else:
        sum_csri_gpu(frontier_nnz_sizes, p.crow_indices(), ones, p.crow_indices().size(0) - 1)
        zero_idxs = next_frontier._indices()[0, (p.values() == 0)].unique()
    zero_count = (next_frontier._indices()[0,:] == 0).nonzero().size(0)
    frontier_nnz_sizes[0] = zero_count

    # frontier_nnz_sizes[zero_idxs] = 0
    frontier_nnz_sizes[zero_idxs] -= 1
    
    timing_dict["sample-pre-loop"].append(stop_time(start_timer, stop_timer))
    sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)

    iter_count = 0
    selection_iter_count = 0
    
    # underfull_minibatches = (sampled_count < frontier_size).any()
    underfull_minibatches = True

    ps_frontier_nnz_sizes = torch.cuda.IntTensor(p.size(0) + 1).fill_(0)
    ps_frontier_nnz_sizes[1:] = torch.cumsum(frontier_nnz_sizes, dtype=torch.int32, dim=0)
    p_rowsum = torch.cuda.DoubleTensor(p.size(0)).fill_(0)
    # p_rowsum = torch.cuda.FloatTensor(p.size(0)).fill_(0)
    while underfull_minibatches:
        iter_count += 1
        start_time(sample_start_timer)

        start_time(start_timer)

        if name == "ladies":
            ps_p_values = torch.cumsum(p._values(), dim=0).roll(1)
            ps_p_values[0] = 0
            p_rowsum.fill_(0)
            p_rowsum.scatter_add_(0, p._indices()[0, :], p._values())
            ps_p_rowsum = torch.cumsum(p_rowsum, dim=0).roll(1)
            ps_p_rowsum[0] = 0
        elif name == "sage":
            ps_p_values = torch.cumsum(p.values(), dim=0)
            p_rowsum.fill_(0)
            # p_rowsum.scatter_add_(0, p._indices()[0, :], p._values())
            sum_csrd_gpu(p_rowsum, p.crow_indices(), p.values(), p.crow_indices().size(0) - 1)
            # ps_p_rowsum = torch.cumsum(p_rowsum, dim=0)
            ps_p_rowsum = torch.cumsum(p_rowsum, dim=0).roll(1)
            ps_p_rowsum[0] = 0
        timing_dict["sample-prob-rowsum"].append(stop_time(start_timer, stop_timer))

        # n_darts_col = int((frontier_size - sampled_count).sum().item() / p.size(0)) * 5 + 1
        rank_batches_start = rank_col * (sampled_count.size(0) // replication)
        rank_batches_stop = (rank_col + 1) * (sampled_count.size(0) // replication)
        rank_batches_stop = min(sampled_count.size(0), rank_batches_stop)
        dart_count_row = frontier_size - sampled_count
        dart_count_row[:rank_batches_start] = 0
        dart_count_row[rank_batches_stop:] = 0
        n_darts_col = dart_count_row.sum().item()

        if n_darts_col > 0:
            start_time(start_timer)
            # dart_values = torch.cuda.DoubleTensor(n_darts * mb_count).uniform_()
            # dart_values = torch.cuda.DoubleTensor(n_darts_col * p.size(0)).uniform_()
            # dart_values = torch.cuda.FloatTensor(n_darts_col * p.size(0)).uniform_()
            # dart_values = torch.cuda.FloatTensor(n_darts_col).uniform_()
            dart_values = torch.cuda.DoubleTensor(n_darts_col).uniform_()
            # ps_dart_count_row = torch.cumsum(dart_count_row, dim=0, dtype=torch.int32).roll(1)
            ps_dart_count_row = torch.cumsum(dart_count_row, dim=0)
            timing_dict["sample-gen-darts"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            # compute_darts1d_gpu(dart_values, n_darts, mb_count)
            # compute_darts1d_gpu(dart_values, n_darts_col, mb_count)
            # compute_darts1d_gpu(dart_values, p_rowsum, ps_p_rowsum, n_darts_col, p.size(0))
            compute_darts1d_gpu(dart_values, p_rowsum, ps_p_rowsum, ps_dart_count_row, n_darts_col, p.size(0))

            timing_dict["sample-dart-throw"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            dart_hits_count = torch.cuda.IntTensor(p._nnz()).fill_(0)
            throw_darts1d_gpu(dart_values, ps_p_values, dart_hits_count, \
                                    n_darts_col, p._nnz())
            timing_dict["sample-filter-darts"].append(stop_time(start_timer, stop_timer))

            frontier_nnz_start = ps_frontier_nnz_sizes[rank_batches_start]
            frontier_nnz_stop = ps_frontier_nnz_sizes[rank_batches_stop]
            dart_hits_count[:frontier_nnz_start] = 0
            dart_hits_count[frontier_nnz_stop:] = 0
        else:
            dart_hits_count = torch.cuda.IntTensor(p._nnz()).fill_(0)

        start_time(start_timer)
        if replication > 1:
            dist.all_reduce(dart_hits_count, group=row_groups[rank_c])
        next_frontier_values = torch.logical_or(
                                    dart_hits_count, 
                                    next_frontier._values().int()).int()
        # next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
        #                                             values=next_frontier_values,
        #                                             size=(p.size(0), node_count_total))
        next_frontier_tmp = sparse_coo_tensor_gpu(next_frontier._indices(), next_frontier_values,
                                                    torch.Size([p.size(0), node_count_total]))
        timing_dict["sample-add-to-frontier"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        # sampled_count = torch.sparse.sum(next_frontier_tmp, dim=1)._values()
        sampled_count_old = sampled_count.clone()
        sampled_count = sampled_count.fill_(0)
        sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)
        if name == "ladies":
            next_frontier_nnzmask = next_frontier_values.nonzero().squeeze()
            next_frontier_nnzvals = next_frontier_values[next_frontier_nnzmask]
            next_frontier_nnzidxs = next_frontier_tmp._indices()[0, next_frontier_nnzmask]
            # sampled_count.scatter_add_(0, next_frontier_tmp._indices()[0, :], next_frontier_values)
            sampled_count.scatter_add_(0, next_frontier_nnzidxs, next_frontier_nnzvals)
        else:
            sum_csri_gpu(sampled_count, p.crow_indices(), next_frontier_values, p.crow_indices().size(0) - 1)
        timing_dict["sample-count-samples"].append(stop_time(start_timer, stop_timer))

        del sampled_count_old

        start_time(start_timer)
        overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
        overflowed_minibatches = (overflow > 0).any()
        timing_dict["sample-select-preproc"].append(stop_time(start_timer, stop_timer))

        start_time(select_start_timer)

        if rank_col == 0 and overflowed_minibatches:
            while overflowed_minibatches:
                start_time(select_iter_start_timer)

                start_time(start_timer)
                selection_iter_count += 1
                ps_overflow = torch.cumsum(overflow, dim=0)
                total_overflow = ps_overflow[-1].item()
                # n_darts_select = total_overflow // replication
                # if rank_col == replication - 1:
                #     n_darts_select = total_overflow - (replication - 1) * n_darts_select
                n_darts_select = total_overflow
                timing_dict["select-psoverflow"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                dart_hits_inv = dart_hits_count.reciprocal().double()
                dart_hits_inv[dart_hits_inv == float("inf")] = 0
                timing_dict["select-reciprocal"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                dart_hits_inv_mtx = sparse_coo_tensor_gpu(next_frontier._indices(), dart_hits_inv,
                                                                torch.Size([p.size(0), node_count_total]))
                timing_dict["select-instmtx"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                dart_hits_inv_sum = torch.cuda.DoubleTensor(p.size(0)).fill_(0)
                dart_hits_inv_nnzidxs = dart_hits_inv.nonzero().squeeze()
                indices_nnz_idxs = next_frontier._indices()[0,dart_hits_inv_nnzidxs]
                dart_hits_inv_sum.scatter_add_(0, indices_nnz_idxs, dart_hits_inv[dart_hits_inv_nnzidxs])
                timing_dict["select-invsum"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                ps_dart_hits_inv_sum = torch.cumsum(dart_hits_inv_sum, dim=0).roll(1)
                ps_dart_hits_inv_sum[0] = 0
                timing_dict["select-psinv"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                # dart_select = torch.cuda.DoubleTensor(total_overflow).uniform_()
                dart_select = torch.cuda.DoubleTensor(n_darts_select).uniform_()
                # Compute darts for selection 
                compute_darts_select_gpu(dart_select, dart_hits_inv_sum, ps_dart_hits_inv_sum, ps_overflow,
                                                p.size(0), n_darts_select)
                timing_dict["select-computedarts"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                # Throw selection darts 
                ps_dart_hits_inv = torch.cumsum(dart_hits_inv, dim=0)
                # throw_darts_select_gpu(dart_select, ps_dart_hits_inv, dart_hits_count, total_overflow,
                throw_darts_select_gpu(dart_select, ps_dart_hits_inv, dart_hits_count, n_darts_select,
                                            next_frontier._nnz())
                timing_dict["select-throwdarts"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                # dist.all_reduce(dart_hits_count, op=dist.ReduceOp.MIN, group=row_groups[rank_c])
                next_frontier_values = torch.logical_or(dart_hits_count, next_frontier._values()).int()
                # next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                #                                                 values=next_frontier_values,
                #                                                 size=(p.size(0), node_count_total))
                next_frontier_tmp = sparse_coo_tensor_gpu(next_frontier._indices(), next_frontier_values,
                                                                torch.Size([p.size(0), node_count_total]))
                timing_dict["select-add-to-frontier"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                sampled_count = torch.cuda.IntTensor(p.size(0)).fill_(0)
                sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)
                # sampled_count.scatter_add_(0, next_frontier_tmp._indices()[0,:], next_frontier_tmp._values())
                next_frontier_nnzvals = next_frontier_tmp._values().nonzero().squeeze()
                next_frontier_nnzidxs = next_frontier_tmp._indices()[0,next_frontier_nnzvals]
                sampled_count.scatter_add_(0, next_frontier_nnzidxs, \
                                                next_frontier_tmp._values()[next_frontier_nnzvals])
                timing_dict["select-samplecount"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
                overflowed_minibatches = (overflow > 0).any()
                timing_dict["select-overflow"].append(stop_time(start_timer, stop_timer))

                timing_dict["select-iter"].append(stop_time(select_iter_start_timer, select_iter_stop_timer))
        else:
            timing_dict["select-psoverflow"] = []
            timing_dict["select-reciprocal"] = []
            timing_dict["select-instmtx"] = []
            timing_dict["select-invsum"] = []
            timing_dict["select-psinv"] = []
            timing_dict["select-computedarts"] = []
            timing_dict["select-throwdarts"] = []
            timing_dict["select-add-to-frontier"] = []
            timing_dict["select-samplecount"] = []
            timing_dict["select-overflow"] = []
            timing_dict["select-iter"] = []

        # dist.barrier(group=row_groups[rank_c])

        timing_dict["sample-dart-selection"].append(stop_time(select_start_timer, select_stop_timer))

        start_time(start_timer)
        if replication > 1:
            dist.all_reduce(dart_hits_count, group=row_groups[rank_c])
            dist.broadcast(dart_hits_count, src=rank_c * replication, group=row_groups[rank_c])
        next_frontier_values = torch.logical_or(
                                        dart_hits_count, 
                                        next_frontier._values()).int()
        # next_frontier = torch.sparse_coo_tensor(indices=next_frontier._indices(),
        #                                         values=next_frontier_values,
        #                                         size=(p.size(0), node_count_total))
        next_frontier = sparse_coo_tensor_gpu(next_frontier._indices(), next_frontier_values,
                                                torch.Size([p.size(0), node_count_total]))
        timing_dict["sample-gather-counts"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        dart_hits_mask = dart_hits_count > 0
        p.values()[dart_hits_mask] = 0.0

        filled_minibatches = sampled_count == frontier_size
        # filled_minibatches_mask = torch.gather(filled_minibatches, 0, p._indices()[0,:])
        filled_minibatches_mask = torch.gather(filled_minibatches, 0, next_frontier._indices()[0,:])
        p.values()[filled_minibatches_mask] = 0
        timing_dict["sample-set-probs"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        sampled_count = torch.cuda.IntTensor(p.size(0)).fill_(0)
        sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)
        if name == "ladies":
            next_frontier_nnzvals = next_frontier._values().nonzero().squeeze()
            next_frontier_nnzidxs = next_frontier._indices()[0,next_frontier_nnzvals]
            sampled_count.scatter_add_(0, next_frontier_nnzidxs, \
                                        next_frontier._values()[next_frontier_nnzvals].int())
        else:
            sum_csri_gpu(sampled_count, p.crow_indices(), next_frontier._values(), p.crow_indices().size(0) - 1)
        underfull_minibatches = (sampled_count < frontier_size).any()
        timing_dict["sample-compute-bool"].append(stop_time(start_timer, stop_timer))

        overflow = torch.clamp(sampled_count - frontier_size, min=0).int()

        timing_dict["sampling-iters"].append(stop_time(sample_start_timer, sample_stop_timer))

    return next_frontier

def select(next_frontier, adj_matrix, batches, sa_masks, nnz, \
                batch_size, frontier_size, mb_count, mb_count_total, node_count_total, replication, \
                rank, size, row_groups, col_groups, timing_dict, layer_id, name, semibulk_size=128):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    semibulk_size = min(semibulk_size, mb_count // replication)

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-construct-nextf")
    if name == "ladies":
        next_frontier_select = torch.masked_select(next_frontier._indices()[1,:], \
                                                next_frontier._values().bool()).view(mb_count, frontier_size)
        batches_select = torch.masked_select(batches._indices()[1,:], \
                                                batches._values().bool()).view(mb_count, batch_size)
    elif name == "sage":
        next_frontier_select = next_frontier._indices()[1,:].view(mb_count * batch_size, frontier_size)
        batches_select = torch.masked_select(batches._indices()[1,:], \
                                                batches._values().bool()).view(mb_count * batch_size, 1)
    next_frontier_select = torch.cat((next_frontier_select, batches_select), dim=1)
    torch.cuda.nvtx.range_pop() # construct-nextf
    timing_dict["construct-nextf"].append(stop_time(start_timer, stop_timer, barrier=True))

    torch.cuda.nvtx.range_push("nvtx-select-rowcols")
    # 1. Make the row/col select matrices
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-select-mtxs")

    if layer_id == 0:
        row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz * mb_count).cuda(),
                                              batches_select.view(-1)))
    else:
        # TODO i > 0 (make sure size args to spspmm are right)
        print("ERROR i > 0")

    row_select_mtx_values = torch.cuda.DoubleTensor(row_select_mtx_indices[0].size(0)).fill_(1.0)

    repeated_next_frontier = next_frontier_select.clone()
    if name == "ladies":
        scale_mtx = torch.arange(start=0, end=node_count_total * mb_count, step=node_count_total).cuda()
    elif name == "sage":
        scale_mtx = torch.arange(start=0, end=node_count_total * mb_count * batch_size, 
                                        step=node_count_total).cuda()
    scale_mtx = scale_mtx[:, None]
    repeated_next_frontier.add_(scale_mtx)
    col_select_mtx_rows = repeated_next_frontier.view(-1)
    if name == "ladies":
        col_select_mtx_cols = torch.arange(next_frontier_select.size(1)).cuda().repeat(
                                        mb_count,1).view(-1)
    elif name == "sage":
        col_select_mtx_cols = torch.arange(next_frontier_select.size(1) * batch_size).cuda().repeat(
                                        mb_count,1).view(-1)
    col_select_mtx_indices = torch.stack((col_select_mtx_rows, col_select_mtx_cols))

    # col_select_mtx_values = torch.cuda.DoubleTensor(col_select_mtx_rows.size(0)).fill_(1.0)

    col_select_mtx_values = torch.cuda.DoubleTensor(col_select_mtx_rows.size(0)).fill_(0.0)
    col_unique_rows, col_inverse = torch.unique(col_select_mtx_rows, sorted=True, return_inverse=True)
    col_rows_perm = torch.arange(col_inverse.size(0), dtype=col_inverse.dtype, device=col_inverse.device)
    col_row_mask = col_inverse.new_empty(col_unique_rows.size(0)).scatter_(0, col_inverse, col_rows_perm)
    col_select_mtx_values[col_row_mask] = 1.0

    torch.cuda.nvtx.range_pop()
    timing_dict["select-mtxs"].append(stop_time(start_timer, stop_timer, barrier=True))

    # 2. Multiply row_select matrix with adj_matrix
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-row-select-spgemm")
    # row_select_mtx = torch.sparse_coo_tensor(row_select_mtx_indices, row_select_mtx_values, 
    #                                                 size=(nnz * mb_count, node_count_total))
    row_select_mtx = sparse_coo_tensor_gpu(row_select_mtx_indices, row_select_mtx_values, 
                                                    torch.Size([nnz * mb_count, node_count_total]))
    if sa_masks is not None:
        sa_masks.fill_(0)
    row_select_mtx = row_select_mtx.to_sparse_csr()
    # adj_matrix = adj_matrix.to_sparse_csr()
    sampled_indices, sampled_values = dist_saspgemm15D(row_select_mtx, adj_matrix, replication, rank, size, \
                                                        row_groups, col_groups, "rowsel", sa_masks, 
                                                        timing_dict)
    row_select_mtx = row_select_mtx.to_sparse_coo()
    # adj_matrix = adj_matrix.to_sparse_coo()
    # sample_mtx = torch.sparse_coo_tensor(sampled_indices, sampled_values, 
    #                                         size=(nnz * mb_count, node_count_total))
    sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                                            torch.Size([nnz * mb_count, node_count_total]))

    torch.cuda.nvtx.range_pop()
    timing_dict["row-select-spgemm"].append(stop_time(start_timer, stop_timer, barrier=True))

    # 3. Expand sampled rows
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-row-select-expand")
    row_shift = torch.cuda.LongTensor(sampled_values.size(0)).fill_(0)
    if name == "ladies":
        # semibulk_size = mb_count // (64 * replication) # number of batches to column extract from in bulk
        # semibulk_size = 4 # number of batches to column extract from in bulk
        # semibulk_size = 128 # number of batches to column extract from in bulk
        shift_rowselect_gpu(row_shift, sampled_indices[0,:], sampled_values.size(0), 
                                rank, size, replication, batch_size, node_count_total, mb_count_total, 
                                batch_size, semibulk_size)
    elif name == "sage":
        semibulk_size = mb_count * nnz # vtxs to column extract from in bulk
        shift_rowselect_gpu(row_shift, sampled_indices[0,:], sampled_values.size(0), 
                                rank, size, replication, batch_size, node_count_total, mb_count_total, 1,
                                semibulk_size)
    sampled_indices[1,:] += row_shift
    torch.cuda.nvtx.range_pop()
    timing_dict["row-select-expand"].append(stop_time(start_timer, stop_timer, barrier=True))

    # 4. Multiply sampled rows with col_select matrix
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-col-select-spgemm")
    if name == "ladies":
        # sample_mtx = torch.sparse_coo_tensor(sampled_indices, sampled_values, 
        #                                         size=(nnz * mb_count, node_count_total * mb_count_total))
        sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                            torch.Size([nnz * mb_count, node_count_total * semibulk_size]))
        # sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
        #                                     torch.Size([nnz * mb_count, node_count_total]))
    elif name == "sage":
        sample_mtx = torch.sparse_coo_tensor(sampled_indices, sampled_values, 
                                                size=(nnz * mb_count, node_count_total * mb_count_total * nnz))
        # sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
        #                                 torch.Size([nnz * mb_count, node_count_total * semibulk_size]))
        # sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
        #                                 torch.Size([nnz * mb_count, node_count_total]))
    if name == "ladies":
        # col_select_mtx = torch.sparse_coo_tensor(col_select_mtx_indices, col_select_mtx_values,
        #                                         size=(node_count_total * mb_count, next_frontier.size(1)))
        # col_select_mtx = sparse_coo_tensor_gpu(col_select_mtx_indices, col_select_mtx_values,
    #                                         torch.Size([node_count_total * mb_count, next_frontier.size(1)]))
        col_select_mtx = sparse_coo_tensor_gpu(col_select_mtx_indices, col_select_mtx_values,
                                        torch.Size([node_count_total * mb_count, next_frontier_select.size(1)]))
    elif name == "sage":
        # col_select_mtx = torch.sparse_coo_tensor(col_select_mtx_indices, col_select_mtx_values,
        #                     size=(node_count_total * mb_count * batch_size, next_frontier.size(1) * batch_size))
        col_select_mtx = torch.sparse_coo_tensor(col_select_mtx_indices, col_select_mtx_values,
                torch.Size([node_count_total * mb_count * batch_size, next_frontier_select.size(1) * batch_size]))

    # sampled_indices, sampled_values = dist_spgemm15D(sample_mtx, col_select_mtx, replication, rank, size, \
    #                                                     row_groups, col_groups, "colsel", timing_dict)

    sample_mtx = sample_mtx.to_sparse_csr()
    # sampled_indices, sampled_values = col_select15D(sample_mtx, next_frontier_select, mb_count, batch_size,
    #                                             replication, rank, size, row_groups, col_groups, "colsel", 
    #                                             timing_dict)
    if name == "ladies":
        bulk_items = mb_count
        bulk_size = batch_size
    elif name == "sage":
        bulk_items = mb_count * nnz
        bulk_size = 1
    sampled_indices, sampled_values = col_select15D(sample_mtx, col_select_mtx, bulk_items, node_count_total, 
                                                bulk_size, semibulk_size, replication, rank, size, row_groups,
                                                col_groups, "colsel", name, timing_dict)

    torch.cuda.nvtx.range_pop()
    timing_dict["col-select-spgemm"].append(stop_time(start_timer, stop_timer, barrier=True))

    # 5. Store sampled matrices
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-set-sample")
    # adj_matrix_sample = torch.sparse_coo_tensor(indices=sampled_indices, values=sampled_values, \
    #                                             size=(nnz * mb_count, next_frontier.size(1)))
    # adj_matrix_sample = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
    #                                         torch.Size([nnz * mb_count, next_frontier.size(1)]))
    adj_matrix_sample = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                                            torch.Size([nnz * mb_count, next_frontier_select.size(1)]))

    torch.cuda.nvtx.range_pop()
    timing_dict["set-sample"].append(stop_time(start_timer, stop_timer, barrier=True))

    torch.cuda.nvtx.range_pop() # nvtx-select-mtxs

    return batches_select, next_frontier_select, adj_matrix_sample

class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preprocess=True,
    ):
        if hparams is None:
            hparams = {}
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage
        self.preprocess = preprocess
        self.transform = transform

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility

    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event_path = self.input_paths[idx]
        print(f"event_path: {event_path}", flush=True)
        # if "event005000555.pyg" in event_path:
        #     return None
        event_name = event_path.split("/")[-1]
        event_id = int(event_name[5:event_name.index(".")])

        # if "CTD" in event_path:
        #     event = torch.load(f"/global/cfs/cdirs/m1982/alokt/data/trackml/ctd/trainset_processed/{event_name}", map_location=torch.device("cpu"))
        #     return event

        event = torch.load(event_path, map_location=torch.device("cpu"))
        # convert DataBatch to Data instance because some transformations don't work on DataBatch
        event = Data(**event.to_dict())
        if not self.preprocess:
            return event
        event = self.preprocess_event(event)
        # do pyg transformation if a torch_geometric.transforms instance is given
        if self.transform is not None:
            event = self.transform(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        event = self.apply_hard_cuts(event)
        event = self.construct_weighting(event)
        event = self.handle_edge_list(event)
        event = self.scale_features(event)
        if self.hparams.get("edge_features") is not None:
            event = self.add_edge_features(
                event
            )  # scaling must be done before adding features
        return event

    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if (
            self.hparams is not None
            and "hard_cuts" in self.hparams.keys()
            and self.hparams["hard_cuts"]
        ):
            assert isinstance(
                self.hparams["hard_cuts"], dict
            ), "Hard cuts must be a dictionary"
            handle_hard_cuts(event, self.hparams["hard_cuts"])

        return event

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """

        assert event.y.shape[0] == event.edge_index.shape[1], (
            f"Input graph has {event.edge_index.shape[1]} edges, but"
            f" {event.y.shape[0]} truth labels"
        )

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(
                self.hparams["weighting"][0], dict
            ), "Weighting must be a list of dictionaries"
            event.weights = handle_weighting(event, self.hparams["weighting"])
        else:
            event.weights = torch.ones_like(event.y, dtype=torch.float32)

        return event

    def handle_edge_list(self, event):
        if (
            "input_cut" in self.hparams.keys()
            and self.hparams["input_cut"]
            and "scores" in event.keys()
        ):
            # Apply a score cut to the event
            self.apply_score_cut(event, self.hparams["input_cut"])

        # if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
        #     # Flip event.edge_index and concat together
        #     self.to_undirected(event)
        return event

    def to_undirected(self, event):
        """
        Add the reverse of the edge_index to the event. This then requires all edge features to be duplicated.
        Additionally, the truth map must be duplicated.
        """
        num_edges = event.edge_index.shape[1]
        # Flip event.edge_index and concat together
        event.edge_index = torch.cat(
            [event.edge_index, event.edge_index.flip(0)], dim=1
        )
        # event.edge_index, unique_edge_indices = torch.unique(event.edge_index, dim=1, return_inverse=True)
        num_track_edges = event.track_edges.shape[1]
        event.track_edges = torch.cat(
            [event.track_edges, event.track_edges.flip(0)], dim=1
        )

        # Concat all edge-like features together
        for key in get_pyg_data_keys(event):
            if key == "truth_map":
                continue
            if not isinstance(event[key], torch.Tensor) or not event[key].shape:
                continue
            if event[key].shape[0] == num_edges:
                event[key] = torch.cat([event[key], event[key]], dim=0)
            elif event[key].shape[0] == num_track_edges:
                event[key] = torch.cat([event[key], event[key]], dim=0)

        # handle truth_map separately
        truth_map = event.truth_map.clone()
        truth_map[truth_map >= 0] = truth_map[truth_map >= 0] + num_edges
        event.truth_map = torch.cat([event.truth_map, truth_map], dim=0)

        return event

    def add_edge_features(self, event):
        if "edge_features" in self.hparams.keys():
            assert isinstance(
                self.hparams["edge_features"], list
            ), "Edge features must be a list of strings"
            handle_edge_features(event, self.hparams["edge_features"])
        return event

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in get_pyg_data_keys(
                    event
                ), f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

        return event

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in get_pyg_data_keys(
                    event
                ), f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]
        return event

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.scores >= score_cut
        num_edges = event.edge_index.shape[1]
        for key in get_pyg_data_keys(event):
            if (
                isinstance(event[key], torch.Tensor)
                and event[key].shape
                and (
                    event[key].shape[0] == num_edges
                    or event[key].shape[-1] == num_edges
                )
            ):
                event[key] = event[key][..., passing_edges_mask]

        remap_from_mask(event, passing_edges_mask)
        return event

    def get_y_node(self, event):
        y_node = torch.zeros(event.z.size(0))
        y_node[event.track_edges.view(-1)] = 1
        event.y_node = y_node
        return event

def load_datafiles_in_dir(input_dir, data_name=None, data_num=None):
    if data_name is not None:
        input_dir = os.path.join(input_dir, data_name)

    data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")][:data_num]
    print(f"data_files: {data_files}", flush=True)
    assert len(data_files) > 0, f"No data files found in {input_dir}"
    if data_num is not None:
        assert len(data_files) == data_num, (
            f"Number of data files found ({len(data_files)}) is less than the number"
            f" requested ({data_num})"
        )

    return data_files

def handle_hard_cuts(event, hard_cuts_config):
    true_track_mask = torch.ones_like(event.truth_map, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        true_track_mask = true_track_mask * value_mask

    graph_mask = torch.isin(
        event.edge_index, event.track_edges[:, true_track_mask]
    ).all(0)
    remap_from_mask(event, graph_mask)

    num_edges = event.edge_index.shape[1]
    for edge_key in get_pyg_data_keys(event):
        if (
            isinstance(event[edge_key], torch.Tensor)
            and num_edges in event[edge_key].shape
        ):
            event[edge_key] = event[edge_key][..., graph_mask]

    num_track_edges = event.track_edges.shape[1]
    for track_feature in get_pyg_data_keys(event):
        if (
            isinstance(event[track_feature], torch.Tensor)
            and num_track_edges in event[track_feature].shape
        ):
            event[track_feature] = event[track_feature][..., true_track_mask]

def reset_angle(angles):
    angles[angles > torch.pi] = angles[angles > torch.pi] - 2 * torch.pi
    angles[angles < -torch.pi] = angles[angles < -torch.pi] + 2 * torch.pi
    return angles

def handle_edge_features(event, edge_features):
    src, dst = event.edge_index

    for edge_feature in edge_features:
        if "dr" in edge_features and not ("dr" in get_pyg_data_keys(event)):
            event.dr = event.r[dst] - event.r[src]
        if "dphi" in edge_features and not ("dphi" in get_pyg_data_keys(event)):
            event.dphi = (
                reset_angle((event.phi[dst] - event.phi[src]) * torch.pi) / torch.pi
            )
        if "dz" in edge_features and not ("dz" in get_pyg_data_keys(event)):
            event.dz = event.z[dst] - event.z[src]
        if "deta" in edge_features and not ("deta" in get_pyg_data_keys(event)):
            event.deta = event.eta[dst] - event.eta[src]
        if "phislope" in edge_features and not ("phislope" in get_pyg_data_keys(event)):
            dr = event.r[dst] - event.r[src]
            dphi = reset_angle((event.phi[dst] - event.phi[src]) * torch.pi) / torch.pi
            phislope = dphi / dr
            event.phislope = phislope
        if "phislope" in edge_features:
            event.phislope = torch.nan_to_num(
                event.phislope, nan=0.0, posinf=100, neginf=-100
            )
            event.phislope = torch.clamp(event.phislope, -100, 100)
        if "rphislope" in edge_features and not (
            "rphislope" in get_pyg_data_keys(event)
        ):
            r_ = (event.r[dst] + event.r[src]) / 2.0
            dr = event.r[dst] - event.r[src]
            dphi = reset_angle((event.phi[dst] - event.phi[src]) * torch.pi) / torch.pi
            phislope = dphi / dr
            phislope = torch.nan_to_num(phislope, nan=0.0, posinf=100, neginf=-100)
            phislope = torch.clamp(phislope, -100, 100)
            rphislope = torch.multiply(r_, phislope)
            event.rphislope = rphislope  # features / norm / pre_proc once
        if "rphislope" in edge_features:
            event.rphislope = torch.nan_to_num(event.rphislope, nan=0.0)

def get_pyg_data_keys(event: Data):
    """
    Get the keys of the pyG data object.
    """
    if torch_geometric.__version__ < "2.4.0":
        return event.keys
    else:
        return event.keys()

def get_condition_lambda(condition_key, condition_val):
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "not_in": lambda event: ~torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "within": lambda event: (condition_val[0] <= event[condition_key].float())
        & (event[condition_key].float() <= condition_val[1]),
        "not_within": lambda event: not (
            (condition_val[0] <= event[condition_key].float())
            & (event[condition_key].float() <= condition_val[1])
        ),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (
            event[condition_key].float() <= condition_val[1]
        )
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")

def remap_from_mask(event, edge_mask):
    """
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    truth_map_to_edges = torch.ones(edge_mask.shape[0], dtype=torch.long) * -1
    truth_map_to_edges = truth_map_to_edges.to(event.truth_map.device)
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(
        event.truth_map.shape[0], device=event.truth_map.device
    )[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map = new_map.to(event.truth_map.device)
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(
        truth_map_to_edges.shape[0], device=truth_map_to_edges.device
    )[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)

def handle_weighting(event, weighting_config):
    """
    Take the specification of the weighting and convert this into float values. The default is:
    - True edges have weight 1.0
    - Negative edges have weight 1.0

    The weighting_config can be used to change this behaviour. For example, we might up-weight target particles - that is edges that pass:
    - y == 1
    - primary == True
    - pt > 1 GeV
    - etc. As desired.

    We can also down-weight (i.e. mask) edges that are true, but not of interest. For example, we might mask:
    - y == 1
    - primary == False
    - pt < 1 GeV
    - etc. As desired.
    """

    # Set the default values, which will be overwritten if specified in the config
    weights = torch.zeros_like(event.y, dtype=torch.float)
    weights[event.y == 0] = 1.0

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        weights[get_weight_mask(event, weight_spec["conditions"])] = weight_val

    return weights

def get_weight_mask(event, weight_conditions):
    graph_mask = torch.ones_like(event.y)

    for condition_key, condition_val in weight_conditions.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        graph_mask = graph_mask * map_tensor_handler(
            value_mask,
            output_type="edge-like",
            num_nodes=event.num_nodes,
            edge_index=event.edge_index,
            truth_map=event.truth_map,
        )

    return graph_mask

def map_tensor_handler(
    input_tensor: torch.Tensor,
    output_type: str,
    input_type: str = None,
    truth_map: torch.Tensor = None,
    edge_index: torch.Tensor = None,
    track_edges: torch.Tensor = None,
    num_nodes: int = None,
    num_edges: int = None,
    num_track_edges: int = None,
    aggr: str = None,
):
    """
    A general function to handle arbitrary maps of one tensor type to another
    Types are "node-like", "edge-like" and "track-like".
    - node-like: The input tensor is of the same size as the 
        number of nodes in the graph
    - edge-like: The input tensor is of the same size as the 
        number of edges in the graph, that is, the *constructed* graph
    - track-like: The input tensor is of the same size as the 
        number of true track edges in the event, that is, the *truth* graph

    To visualize:
                    (n)
                     ^
                    / \
      edge_to_node /   \ track_to_node
                  /     \
                 /       \
                /         \
               /           \
              /             \
node_to_edge /               \ node_to_track
            /                 \
           v     edge_to_track v
          (e) <-------------> (t)
            track_to_edge

    Args:
        input_tensor (torch.Tensor): The input tensor to be mapped
        output_type (str): The type of the output tensor. 
            One of "node-like", "edge-like" or "track-like"
        input_type (str, optional): The type of the input tensor. 
            One of "node-like", "edge-like" or "track-like". Defaults to None,
            and will try to infer the type from the input tensor, if num_nodes
            and/or num_edges are provided.
        truth_map (torch.Tensor, optional): The truth map tensor. 
            Defaults to None. Used for mappings to/from track-like tensors.
        num_nodes (int, optional): The number of nodes in the graph. 
            Defaults to None. Used for inferring the input type.
        num_edges (int, optional): The number of edges in the graph. 
            Defaults to None. Used for inferring the input type.
        num_track_edges (int, optional): The number of track edges in the graph 
            Defaults to None. Used for inferring the input type.
    """

    if num_track_edges is None and truth_map is not None:
        num_track_edges = truth_map.shape[0]
    if num_track_edges is None and track_edges is not None:
        num_track_edges = track_edges.shape[1]
    if num_edges is None and edge_index is not None:
        num_edges = edge_index.shape[1]
    if input_type is None:
        input_type, input_tensor = infer_input_type(
            input_tensor, num_nodes, num_edges, num_track_edges
        )
    if input_type == output_type:
        return input_tensor

    input_args = {
        "truth_map": truth_map,
        "edge_index": edge_index,
        "track_edges": track_edges,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_track_edges": num_track_edges,
        "aggr": aggr,
    }

    mapping_functions = {
        ("node-like", "edge-like"): map_nodes_to_edges,
        ("edge-like", "node-like"): map_edges_to_nodes,
        ("node-like", "track-like"): map_nodes_to_tracks,
        ("track-like", "node-like"): map_tracks_to_nodes,
        ("edge-like", "track-like"): map_edges_to_tracks,
        ("track-like", "edge-like"): map_tracks_to_edges,
    }
    if (input_type, output_type) not in mapping_functions:
        raise ValueError(f"Mapping from {input_type} to {output_type} not supported")

    return mapping_functions[(input_type, output_type)](input_tensor, **input_args)

# Returns string and tensor
def infer_input_type(
    input_tensor: torch.Tensor,
    num_nodes: int = None,
    num_edges: int = None,
    num_track_edges: int = None,
) -> (str, torch.Tensor):
    """
    Infers the type of the input tensor based on its shape and the provided number of nodes, edges, and track edges.

    Args:
        input_tensor (torch.Tensor): The tensor whose type needs to be inferred.
        num_nodes (int, optional): Number of nodes in the graph.
        num_edges (int, optional): Number of edges in the graph.
        num_track_edges (int, optional): Number of track edges in the graph.

    Returns:
        str: The inferred type of the input tensor. One of ["node-like", "edge-like", "track-like"].
    """

    NODE_LIKE = "node-like"
    EDGE_LIKE = "edge-like"
    TRACK_LIKE = "track-like"

    if num_nodes is not None and input_tensor.shape[0] == num_nodes:
        return NODE_LIKE, input_tensor
    elif num_edges is not None and num_edges in input_tensor.shape:
        return EDGE_LIKE, input_tensor
    elif num_track_edges is not None and num_track_edges in input_tensor.shape:
        return TRACK_LIKE, input_tensor
    elif num_track_edges is not None and num_track_edges // 2 in input_tensor.shape:
        return TRACK_LIKE, torch.cat([input_tensor, input_tensor], dim=0)
    else:
        raise ValueError("Unable to infer the type of the input tensor.")

def map_nodes_to_edges(
    nodelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None, **kwargs
):
    """
    Map a node-like tensor to an edge-like tensor. If the aggregation is None, this is simply done by sending node values to the edges, thus returning a tensor of shape (2, num_edges).
    If the aggregation is not None, the node values are aggregated to the edges, and the resulting tensor is of shape (num_edges,).
    """

    if aggr is None:
        return nodelike_input[edge_index]

    edgelike_tensor = nodelike_input[edge_index]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(edgelike_tensor, dim=0)

def map_edges_to_nodes(
    edgelike_input: torch.Tensor,
    edge_index: torch.Tensor,
    aggr: str = None,
    num_nodes: int = None,
    **kwargs,
):
    """
    Map an edge-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending edge values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the edge values are aggregated to the nodes at the destination node (edge_index[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(
            num_nodes, dtype=edgelike_input.dtype, device=edgelike_input.device
        )
        nodelike_output[edge_index] = edgelike_input
        return nodelike_output

    return scatter(
        edgelike_input, edge_index[1], dim=0, dim_size=num_nodes, reduce=aggr
    )

def map_nodes_to_tracks(
    nodelike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None, **kwargs
):
    """
    Map a node-like tensor to a track-like tensor. If the aggregation is None, this is simply done by sending node values to the tracks, thus returning a tensor of shape (2, num_track_edges).
    If the aggregation is not None, the node values are aggregated to the tracks, and the resulting tensor is of shape (num_track_edges,).
    """

    if aggr is None:
        return nodelike_input[track_edges]

    tracklike_tensor = nodelike_input[track_edges]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(tracklike_tensor, dim=0)

def map_tracks_to_nodes(
    tracklike_input: torch.Tensor,
    track_edges: torch.Tensor,
    aggr: str = None,
    num_nodes: int = None,
    **kwargs,
):
    """
    Map a track-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending track values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the track values are aggregated to the nodes at the destination node (track_edges[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(track_edges.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(
            num_nodes, dtype=tracklike_input.dtype, device=tracklike_input.device
        )
        nodelike_output[track_edges] = tracklike_input
        return nodelike_output

    return scatter(
        tracklike_input.repeat(2),
        torch.cat([track_edges[0], track_edges[1]]),
        dim=0,
        dim_size=num_nodes,
        reduce=aggr,
    )

def map_tracks_to_edges(
    tracklike_input: torch.Tensor,
    truth_map: torch.Tensor,
    num_edges: int = None,
    **kwargs,
):
    """
    Map an track-like tensor to a edge-like tensor. This is done by sending the track value through the truth map, where the truth map is >= 0. Note that where truth_map == -1,
    the true edge has not been constructed in the edge_index. In that case, the value is set to NaN.
    """

    if num_edges is None:
        num_edges = int(truth_map.max().item() + 1)
    edgelike_output = torch.zeros(
        num_edges, dtype=tracklike_input.dtype, device=tracklike_input.device
    )
    edgelike_output[truth_map[truth_map >= 0]] = tracklike_input[truth_map >= 0]
    edgelike_output[truth_map[truth_map == -1]] = float("nan")
    return edgelike_output


def map_edges_to_tracks(
    edgelike_input: torch.Tensor, truth_map: torch.Tensor, **kwargs
):
    """
    TODO: Implement this. I don't think it is a meaningful operation, but it is needed for completeness.
    """
    raise NotImplementedError(
        "This is not a meaningful operation, but it is needed for completeness"
    )

def rearrange_by_distance(event, edge_index):
    assert "r" in get_pyg_data_keys(event) and "z" in get_pyg_data_keys(
        event
    ), "event must contain r and z"
    distance = event.r**2 + event.z**2

    # flip edges that are pointing inward
    edge_mask = distance[edge_index[0]] > distance[edge_index[1]]
    edge_index[:, edge_mask] = edge_index[:, edge_mask].flip(0)

    return edge_index

def graph_roc_curve(dataset_name, test_batches, title, filename):
    """
    Plot the ROC curve for the graph construction efficiency.
    """
    print(
        "Plotting the ROC curve and score distribution, events from"
        f" {dataset_name}"
    )
    all_y_truth, all_scores, masked_scores, masked_y_truth = [], [], [], []
    masks = []
    # dataset_name = config["dataset"]
    # dataset = getattr(lightning_module, dataset_name)

    # for event in tqdm(dataset):
    for event in tqdm(test_batches):
        # event = event.to(lightning_module.device)
        event = event.cuda()
        # Need to apply score cut and remap the truth_map
        if "weights" in get_pyg_data_keys(event):
            target_y = event.weights.bool() & event.y.bool()
            mask = event.weights > 0
        else:
            target_y = event.y.bool()
            mask = torch.ones_like(target_y).bool().to(target_y.device)

        all_y_truth.append(target_y)
        all_scores.append(event.scores)
        masked_scores.append(event.scores[mask])
        masked_y_truth.append(target_y[mask])
        masks.append(mask)

    all_scores = torch.cat(all_scores).cpu().numpy()
    all_y_truth = torch.cat(all_y_truth).cpu().numpy()
    masked_scores = torch.cat(masked_scores).cpu().numpy()
    masked_y_truth = torch.cat(masked_y_truth).cpu().numpy()
    masks = torch.cat(masks).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Get the ROC curve
    fpr, tpr, _ = roc_curve(all_y_truth, all_scores)
    full_auc_score = auc(fpr, tpr)

    # Plot the ROC curve
    ax.plot(fpr, tpr, color="black", label="ROC curve")

    # Get the ROC curve
    fpr, tpr, _ = roc_curve(masked_y_truth, masked_scores)
    masked_auc_score = auc(fpr, tpr)

    # Plot the ROC curve
    ax.plot(fpr, tpr, color="green", label="masked ROC curve")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("True Positive Rate", ha="right", y=0.95, fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=14)
    ax.text(
        0.95,
        0.20,
        f"Full AUC: {full_auc_score:.3f}, Masked AUC: {masked_auc_score: .3f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )

    # Save the plot
    atlasify(
        "Internal",
        # f"{plot_config['title']} \n"
        f"{title} \n"
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$GeV, $|\eta| < 4$"
        + "\n"
        # + f"Evaluated on {dataset.len()} events in {dataset_name}",
        + f"Evaluated on {len(test_batches)} events in {dataset_name}",
    )
    # filename_template = plot_config.get("filename")
    filename_template = filename
    filename = (
        f"{filename_template}_roc_curve.png"
        if filename_template is not None
        else "roc_curve.png"
    )
    # filename = os.path.join(config["stage_dir"], filename)
    filename = os.path.join(".", filename)
    fig.savefig(filename)
    print("Finish plotting. Find the ROC curve at" f" {filename}")
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    all_y_truth = all_y_truth.astype(np.int16)
    all_y_truth[~masks] = 2
    labels = np.array(["Fake"] * len(all_y_truth))
    labels[all_y_truth == 1] = "Target True"
    labels[all_y_truth == 2] = "Non-target True"
    # weight = 1 / dataset.len()
    weight = 1 / len(test_batches)
    ax = plot_score_histogram(all_scores, labels, ax=ax, inverse_dataset_length=weight)
    ax.set_xlabel("Edge score", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Count/event", ha="right", y=0.95, fontsize=14)
    atlasify(
        "Internal",
        "Score Distribution \n"
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$GeV, $|\eta| < 4$"
        + "\n"
        # + f"Evaluated on {dataset.len()} events in {dataset_name}",
        + f"Evaluated on {len(test_batches)} events in {dataset_name}",
    )
    filename = (
        f"{filename_template}_score_distribution.png"
        if filename_template is not None
        else "score_distribution.png"
    )
    # filename = os.path.join(config["stage_dir"], filename)
    filename = os.path.join(".", filename)
    fig.savefig(filename)
    print("Finish plotting. Find the score distribution at" f" {filename}")

    return full_auc_score, masked_auc_score

def graph_intersection(
    input_pred_graph,
    input_truth_graph,
    return_y_pred=True,
    return_y_truth=False,
    return_pred_to_truth=False,
    return_truth_to_pred=False,
    unique_pred=True,
    unique_truth=True,
):
    """
    An updated version of the graph intersection function, which is around 25x faster than the
    Scipy implementation (on GPU). Takes a prediction graph and a truth graph, assumed to have unique entries.
    If unique_pred or unique_truth is False, the function will first find the unique entries in the input graphs, and return the updated edge lists.
    """

    if not unique_pred:
        input_pred_graph = torch.unique(input_pred_graph, dim=1)
    if not unique_truth:
        input_truth_graph = torch.unique(input_truth_graph, dim=1)

    unique_edges, inverse = torch.unique(
        torch.cat([input_pred_graph, input_truth_graph], dim=1),
        dim=1,
        sorted=False,
        return_inverse=True,
        return_counts=False,
    )

    inverse_pred_map = torch.ones_like(unique_edges[1]) * -1
    inverse_pred_map[inverse[: input_pred_graph.shape[1]]] = torch.arange(
        input_pred_graph.shape[1], device=input_pred_graph.device
    )

    inverse_truth_map = torch.ones_like(unique_edges[1]) * -1
    inverse_truth_map[inverse[input_pred_graph.shape[1] :]] = torch.arange(
        input_truth_graph.shape[1], device=input_truth_graph.device
    )

    pred_to_truth = inverse_truth_map[inverse][: input_pred_graph.shape[1]]
    truth_to_pred = inverse_pred_map[inverse][input_pred_graph.shape[1] :]

    return_tensors = []

    if not unique_pred:
        return_tensors.append(input_pred_graph)
    if not unique_truth:
        return_tensors.append(input_truth_graph)
    if return_y_pred:
        y_pred = pred_to_truth >= 0
        return_tensors.append(y_pred)
    if return_y_truth:
        y_truth = truth_to_pred >= 0
        return_tensors.append(y_truth)
    if return_pred_to_truth:
        return_tensors.append(pred_to_truth)
    if return_truth_to_pred:
        return_tensors.append(truth_to_pred)

    return return_tensors if len(return_tensors) > 1 else return_tensors[0]

def apply_score_cut(event, score_cut):
    """
    Apply a score cut to the event. This is used for the evaluation stage.
    """
    passing_edges_mask = event.scores >= score_cut

    # flip edge direction if points inward
    event.edge_index = rearrange_by_distance(event, event.edge_index)
    event.track_edges = rearrange_by_distance(event, event.track_edges)

    event.graph_truth_map = graph_intersection(
        event.edge_index,
        event.track_edges,
        return_y_pred=False,
        return_y_truth=False,
        return_truth_to_pred=True,
    )
    event.truth_map = graph_intersection(
        event.edge_index[:, passing_edges_mask],
        event.track_edges,
        return_y_pred=False,
        return_truth_to_pred=True,
    )
    event.pred = passing_edges_mask

def apply_target_conditions(event, target_tracks):
    """
    Apply the target conditions to the event. This is used for the evaluation stage.
    Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
    """
    # passing_tracks = torch.ones(event.truth_map.shape[0], dtype=torch.bool).to(
    #     self.device
    # )
    passing_tracks = torch.ones(event.truth_map.shape[0], dtype=torch.bool).cuda()

    for condition_key, condition_val in target_tracks.items():
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        # passing_tracks = passing_tracks * condition_lambda(event).to(self.device)
        passing_tracks = passing_tracks * condition_lambda(event).cuda()

    event.target_mask = passing_tracks

def gnn_efficiency_rz(test_batches, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot GNN edgewise efficiency against rz
    """

    print("Plotting edgewise efficiency as a function of rz")
    print(
        f"Using score cut: {config.get('score_cut')}, events from {config['dataset']}"
    )
    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    target = {"z": torch.empty(0), "r": torch.empty(0)}
    all_target = target.copy()
    true_positive = target.copy()
    input_graph_size, graph_size, n_graphs = (0, 0, 0)

    # dataset_name = config["dataset"]
    # dataset = getattr(lightning_module, dataset_name)

    # for event in tqdm(dataset):
    for event in tqdm(test_batches):
        # event = event.to(lightning_module.device)
        event = event.cuda()
        print(f"event.r: {event.r}", flush=True)
        print(f"event.z: {event.z}", flush=True)

        # Need to apply score cut and remap the truth_map
        if "score_cut" in config:
            # lightning_module.apply_score_cut(event, config["score_cut"])
            apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        # scale r and z
        event.r /= 1000
        event.z /= 1000

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)
        event = event.cpu()

        # indices of all target edges present in the input graph
        print(f"event.target_mask: {event.target_mask}", flush=True)
        print(f"event.graph_truth_map: {event.graph_truth_map}", flush=True)
        target_edges = event.track_edges[
            :, event.target_mask & (event.graph_truth_map > -1)
        ]

        # indices of all target edges (may or may not be present in the input graph)
        all_target_edges = event.track_edges[:, event.target_mask]

        # get target z r
        for key, item in target.items():
            target[key] = torch.cat([item, event[key][target_edges[0]]], dim=0)
        for key, item in all_target.items():
            all_target[key] = torch.cat([item, event[key][all_target_edges[0]]], dim=0)

        # indices of all true positive target edges
        target_true_positive_edges = event.track_edges[
            :, event.target_mask & (event.truth_map > -1)
        ]
        for key in ["r", "z"]:
            true_positive[key] = torch.cat(
                [true_positive[key], event[key][target_true_positive_edges[0]]], dim=0
            )

        input_graph_size += event.edge_index.size(1)
        graph_size += event.pred.sum().numpy()
        n_graphs += 1
        event.r *= 1000 # rescale because events test_batches are maintained across function calls
        event.z *= 1000

    print(f"plot_efficiency_rz1", flush=True)
    fig, ax = plot_efficiency_rz(
        target["z"], target["r"], true_positive["z"], true_positive["r"], plot_config
    )
    # Save the plot
    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$ " + "\n"
        "Graph Construction Efficiency:"
        f" {(target['z'].shape[0] / all_target['z'].shape[0]):.4f}, Input graph size:"
        f" {input_graph_size / n_graphs: .2e} \n"
        r"Edge score cut: "
        + str(config["score_cut"])
        + f", Mean graph size: {graph_size / n_graphs :.2e} \n"
        "Signal Efficiency:"
        f" {true_positive['z'].shape[0] / target['z'].shape[0] :.4f} \n"
        "Cumulative signal efficiency:"
        f" {true_positive['z'].shape[0] / all_target['z'].shape[0]: .4f}"
        + "\n"
        # + f"Evaluated on {dataset.len()} events in {dataset_name}",
        + f"Evaluated on {len(test_batches)} events",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        f"{plot_config.get('filename', 'edgewise_efficiency_rz')}.png",
    )
    fig.savefig(save_dir)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()

    print(f"plot_efficiency_rz2", flush=True)
    fig, ax = plot_efficiency_rz(
        all_target["z"],
        all_target["r"],
        true_positive["z"],
        true_positive["r"],
        plot_config,
    )
    # Save the plot
    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$ " + "\n"
        "Graph Construction Efficiency:"
        f" {(target['z'].shape[0] / all_target['z'].shape[0]):.4f}, Input graph size:"
        f" {input_graph_size / n_graphs: .2e} \n"
        r"Edge score cut: "
        + str(config["score_cut"])
        + f", Mean graph size: {graph_size / n_graphs :.2e} \n"
        "Signal Efficiency:"
        f" {true_positive['z'].shape[0] / target['z'].shape[0] :.4f} \n"
        "Cumulative signal efficiency:"
        f" {true_positive['z'].shape[0] / all_target['z'].shape[0]: .4f}"
        + "\n"
        # + f"Evaluated on {dataset.len()} events in {dataset_name}",
        + f"Evaluated on {len(test_batches)} events",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        f"cumulative_{plot_config.get('filename', 'edgewise_efficiency_rz')}.png",
    )
    fig.savefig(save_dir)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()

    signal_efficiency = true_positive['z'].shape[0] / target['z'].shape[0]
    return signal_efficiency

def gnn_purity_rz(test_batches, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot GNN edgewise efficiency against rz
    """

    print("Plotting edgewise purity as a function of rz")
    print(
        f"Using score cut: {config.get('score_cut')}, events from {config['dataset']}"
    )
    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    true_positive = {
        # key: torch.empty(0).to(lightning_module.device) for key in ["z", "r"]
        key: torch.empty(0).cuda() for key in ["z", "r"]
    }
    target_true_positive = true_positive.copy()

    pred = true_positive.copy()
    masked_pred = true_positive.copy()

    # dataset_name = config["dataset"]
    # dataset = getattr(lightning_module, dataset_name)

    # for event in tqdm(dataset):
    for event in tqdm(test_batches):
        # event = event.to(lightning_module.device)
        event = event.cuda()
        # Need to apply score cut and remap the truth_map
        if "score_cut" in config:
            apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        # scale r and z
        event.r /= 1000
        event.z /= 1000
        print(f"event: {event}", flush=True)
        print(f"event.r: {event.r}", flush=True)
        print(f"event.z: {event.z}", flush=True)

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)
        # event = event.cpu()

        # target true positive edge indices, used as numerator of target purity and purity
        target_true_positive_edges = event.track_edges[
            :, event.target_mask & (event.truth_map > -1)
        ]

        # true positive edge indices, used as numerator of total purity
        true_positive_edges = event.track_edges[:, (event.truth_map > -1)]

        # all positive edges, used as denominator of total and target purity
        positive_edges = event.edge_index[:, event.pred]

        # masked positive edge indices, including true positive target edges and all false positive edges
        fake_positive_edges = event.edge_index[:, event.pred & (event.y == 0)]
        masked_positive_edges = torch.cat(
            [target_true_positive_edges, fake_positive_edges], dim=1
        )

        for key in ["r", "z"]:
            target_true_positive[key] = torch.cat(
                [
                    target_true_positive[key].float(),
                    event[key][target_true_positive_edges[0]].float(),
                ],
                dim=0,
            )
            true_positive[key] = torch.cat(
                [
                    true_positive[key].float(),
                    event[key][true_positive_edges[0]].float(),
                ],
                dim=0,
            )
            pred[key] = torch.cat(
                [pred[key].float(), event[key][positive_edges[0]].float()], dim=0
            )
            masked_pred[key] = torch.cat(
                [
                    masked_pred[key].float(),
                    event[key][masked_positive_edges[0]].float(),
                ],
                dim=0,
            )
        print(f"true_positive[z]: {true_positive['z']}", flush=True)
        print(f"true_positive[r]: {true_positive['r']}", flush=True)
        event.r *= 1000 # rescale because events test_batches are maintained across function calls
        event.z *= 1000

    purity_definition_label = {
        "target_purity": "Target Purity",
        "masked_purity": "Masked Purity",
        "total_purity": "Total Purity",
    }
    for numerator, denominator, suffix in zip(
        [true_positive, target_true_positive, target_true_positive],
        [pred, pred, masked_pred],
        ["total_purity", "target_purity", "masked_purity"],
    ):
        print(f"plot_efficiency_rz3", flush=True)
        print(f"numerator[z]: {numerator['z']}", flush=True)
        print(f"numerator[r]: {numerator['r']}", flush=True)
        fig, ax = plot_efficiency_rz(
            denominator["z"].cpu(),
            denominator["r"].cpu(),
            numerator["z"].cpu(),
            numerator["r"].cpu(),
            plot_config,
        )
        # Save the plot
        atlasify(
            "Internal",
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
            r" \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: "
            + str(config["score_cut"])
            + "\n"
            + purity_definition_label[suffix]
            + ": "
            + f"{numerator['z'].size(0) / denominator['z'].size(0) : .5f}"
            + "\n"
            # + f"Evaluated on {dataset.len()} events in {dataset_name}",
            + f"Evaluated on {len(test_batches)} events",
        )
        plt.tight_layout()
        save_dir = os.path.join(
            f"{plot_config.get('filename', 'edgewise')}_{suffix}_rz.png",
        )
        fig.savefig(save_dir)
        print(f"Finish plotting. Find the plot at {save_dir}")
        plt.close()

def plot_efficiency_rz(
    target_z: torch.Tensor,
    target_r: torch.Tensor,
    true_positive_z: torch.Tensor,
    true_positive_r: torch.Tensor,
    plot_config: dict,
):
    z_range, r_range = plot_config.get("z_range", [-3, 3]), plot_config.get(
        "r_range", [0, 1.0]
    )
    z_bins, r_bins = plot_config.get("z_bins", 6 * 64), plot_config.get("r_bins", 64)
    z_bins = np.linspace(z_range[0], z_range[1], z_bins, endpoint=True)
    r_bins = np.linspace(r_range[0], r_range[1], r_bins, endpoint=True)

    fig, ax = plt.subplots(1, 1, figsize=plot_config.get("fig_size", (12, 6)))
    true_hist, _, _ = np.histogram2d(
        target_z.numpy(),
        target_r.numpy(),
        bins=[z_bins, r_bins],
    )
    true_positive_hist, z_edges, r_edges = np.histogram2d(
        true_positive_z.numpy(), true_positive_r.numpy(), bins=[z_bins, r_bins]
    )

    print(f"target_z: {target_z} z_bins: {z_bins}", flush=True)
    print(f"target_r: {target_r} r_bins: {r_bins}", flush=True)
    print(f"true_positive_hist: {true_positive_hist}", flush=True)
    print(f"true_hist: {true_hist}", flush=True)
    eff = true_positive_hist / true_hist
    print(f"eff.T: {eff.T}", flush=True)
    print(f"eff.T.nonnans: {np.count_nonzero(~np.isnan(eff.T))} eff.T.numelems: {eff.T.size}", flush=True)

    c = ax.pcolormesh(
        z_bins,
        r_bins,
        eff.T,
        cmap="jet_r",
        vmin=plot_config.get("vmin", 0.9),
        vmax=plot_config.get("vmax", 1),
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [m]")

    return fig, ax

def plot_score_histogram(scores, y, bins=100, ax=None, inverse_dataset_length=1):
    """
    Plot a histogram of scores, labelled by truth
    """
    # weight each entry by the inverse of dataset length, such that the y axis relects the count per event per bin
    weights = np.array([inverse_dataset_length] * len(scores))
    ax = sns.histplot(
        x=scores,
        hue=y,
        bins=100,
        stat="count",
        weights=weights,
        log_scale=(False, True),
        ax=ax,
        element="step",
        palette="colorblind",
        fill=False,
    )
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=np.unique(y).shape[0],
        title=None,
        frameon=False,
    )

    return ax

# class GraphDataset(Dataset):
#     """
#     The custom default GNN dataset to load graphs off the disk
#     """
# 
#     def __init__(self, input_dir, data_name = None, num_events = None, stage="fit", hparams=None, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(input_dir, transform, pre_transform, pre_filter)
#         
#         self.input_dir = input_dir
#         self.data_name = data_name
#         self.hparams = hparams
#         self.num_events = num_events
#         self.stage = stage
#         
#         self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
#         self.input_paths.sort() # We sort here for reproducibility
#         
#     def len(self):
#         return len(self.input_paths)
# 
#     def get(self, idx):
# 
#         event_path = self.input_paths[idx]
#         event = torch.load(event_path, map_location=torch.device("cpu"))
#         #del event.track_edges
#         #event.track_edges=torch.t(event.track_edges)
#         #print("Event: ", event)
#         self.preprocess_event(event)
#         #print(self.hparams["node_features"]) ['r', 'phi', 'z']
#         #event["x"] = torch.stack([event[feature] for feature in self.hparams["node_features"]], dim=-1).float()
#         #event = Data(x=event['x'], edge_index=event['edge_index'],track_edges=event["track_edges"], truth_map=event["truth_map"], y=event["y"])
#         #print("Event:", event_path, event)
#         # return (event, event_path) if self.stage == "predict" else event
#         return event
# 
#     def preprocess_event(self, event):
#         """
#         Process event before it is used in training and validation loops
#         """
#         
#         print(f"preprocessing event", flush=True)
#         self.apply_hard_cuts(event)
#         self.construct_weighting(event)
#         self.handle_edge_list(event)
#         self.add_edge_features(event)
#         self.scale_features(event)
#         
#     def apply_hard_cuts(self, event):
#         """
#         Apply hard cuts to the event. This is implemented by 
#         1. Finding which true edges are from tracks that pass the hard cut.
#         2. Pruning the input graph to only include nodes that are connected to these edges.
#         """
#         
#         if self.hparams is not None and "hard_cuts" in self.hparams.keys() and self.hparams["hard_cuts"]:
#             assert isinstance(self.hparams["hard_cuts"], dict), "Hard cuts must be a dictionary"
#             handle_hard_cuts(event, self.hparams["hard_cuts"])
# 
#     def construct_weighting(self, event):
#         """
#         Construct the weighting for the event
#         """
#         
#         assert event.y.shape[0] == event.edge_index.shape[1], f"Input graph has {event.edge_index.shape[1]} edges, but {event.y.shape[0]} truth labels"
# 
#         if self.hparams is not None and "weighting" in self.hparams.keys():
#             assert isinstance(self.hparams["weighting"], list) & isinstance(self.hparams["weighting"][0], dict), "Weighting must be a list of dictionaries"
#             event.weights = handle_weighting(event, self.hparams["weighting"])
#         else:
#             event.weights = torch.ones_like(event.y, dtype=torch.float32)
#             
#     def handle_edge_list(self, event):
# 
#         if "input_cut" in self.hparams.keys() and self.hparams["input_cut"] and "scores" in event.keys:
#             # Apply a score cut to the event
#             self.apply_score_cut(event, self.hparams["input_cut"])
# 
#         # if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
#         #     # Flip event.edge_index and concat together
#         #     self.to_undirected(event)
#             
#     
#     def to_undirected(self, event):
#         """
#         Add the reverse of the edge_index to the event. This then requires all edge features to be duplicated.
#         Additionally, the truth map must be duplicated.
#         """
# 
#         num_edges = event.edge_index.shape[1]
#         # Flip event.edge_index and concat together
#         event.edge_index = torch.cat([event.edge_index, event.edge_index.flip(0)], dim=1)
#         # event.edge_index, unique_edge_indices = torch.unique(event.edge_index, dim=1, return_inverse=True)
# 
#         # Concat all edge-like features together
#         for key in event.keys:
#             if isinstance(event[key], torch.Tensor) and ((event[key].shape[0] == num_edges)):
#                 event[key] = torch.cat([event[key], event[key]], dim=0)
#                 # event[key] = torch.zeros_like(event.edge_index[0], dtype=event[key].dtype).scatter(0, unique_edge_indices, event[key])
# 
# 
#     def add_edge_features(self, event):
#         if "edge_features" in self.hparams.keys():
#             assert isinstance(self.hparams["edge_features"], list), "Edge features must be a list of strings"
#             handle_edge_features(event, self.hparams["edge_features"])
# 
#     def scale_features(self, event):
#         """
#         Handle feature scaling for the event
#         """
#         
#         print(f"hparams: {self.hparams}", flush=True)
#         if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
#             assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
#             for i, feature in enumerate(self.hparams["node_features"]):
#                 assert feature in event.keys, f"Feature {feature} not found in event"
#                 event[feature] = event[feature] / self.hparams["node_scales"][i]
#  
#     def unscale_features(self, event):
#         """
#         Unscale features when doing prediction
#         """
#         
#         if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
#             assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
#             for i, feature in enumerate(self.hparams["node_features"]):
#                 assert feature in event.keys, f"Feature {feature} not found in event"
#                 event[feature] = event[feature] * self.hparams["node_scales"][i]
# 
#     def apply_score_cut(self, event, score_cut):
#         """
#         Apply a score cut to the event. This is used for the evaluation stage.
#         """
#         passing_edges_mask = event.scores >= score_cut
#         num_edges = event.edge_index.shape[1]
#         for key in event.keys:
#             if isinstance(event[key], torch.Tensor) and event[key].shape and (event[key].shape[0] == num_edges or event[key].shape[-1] == num_edges):
#                 event[key] = event[key][..., passing_edges_mask]
# 
#         remap_from_mask(event, passing_edges_mask)
# 
# def load_datafiles_in_dir(input_dir, data_name = None, data_num = None):
# 
#     if data_name is not None:
#         input_dir = os.path.join(input_dir, data_name)
# 
#     data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")][:data_num]
#     assert len(data_files) > 0, f"No data files found in {input_dir}"
#     if data_num is not None:
#         assert len(data_files) == data_num, f"Number of data files found ({len(data_files)}) is less than the number requested ({data_num})"
# 
#     return data_files
# 
# def handle_hard_cuts(event, hard_cuts_config):
# 
#     true_track_mask = torch.ones_like(event.truth_map, dtype=torch.bool)
# 
#     print(f"config: {hard_cuts_config}", flush=True)
#     for condition_key, condition_val in hard_cuts_config.items():
#         assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
#         condition_lambda = get_condition_lambda(condition_key, condition_val)
#         value_mask = condition_lambda(event)
#         true_track_mask = true_track_mask * value_mask
# 
#     graph_mask = torch.isin(event.edge_index, event.track_edges[:, true_track_mask]).all(0)
#     remap_from_mask(event, graph_mask)
# 
#     num_edges = event.edge_index.shape[1]
#     for edge_key in event.keys:
#         if isinstance(event[edge_key], torch.Tensor) and num_edges in event[edge_key].shape:
#             event[edge_key] = event[edge_key][..., graph_mask]
# 
#     num_track_edges = event.track_edges.shape[1]
#     for track_feature in event.keys:
#         if isinstance(event[track_feature], torch.Tensor) and num_track_edges in event[track_feature].shape:
#             event[track_feature] = event[track_feature][..., true_track_mask]
# 
# def get_condition_lambda(condition_key, condition_val):
# 
#     condition_dict = {
#         "is": lambda event: event[condition_key] == condition_val,
#         "is_not": lambda event: event[condition_key] != condition_val,
#         "in": lambda event: torch.isin(event[condition_key], torch.tensor(condition_val[1], device=event[condition_key].device)),
#         "not_in": lambda event: ~torch.isin(event[condition_key], torch.tensor(condition_val[1], device=event[condition_key].device)),
#         "within": lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1]),
#         "not_within": lambda event: not ((condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])),
#     }
# 
#     if isinstance(condition_val, bool):
#         return lambda event: event[condition_key] == condition_val
#     elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
#         return lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])
#     elif isinstance(condition_val, list):
#         return condition_dict[condition_val[0]]
#     else:
#         raise ValueError(f"Condition {condition_val} not recognised")
# 
# def remap_from_mask(event, edge_mask):
#     """ 
#     Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
#     """
# 
#     truth_map_to_edges = torch.ones(edge_mask.shape[0], dtype=torch.long) * -1
#     truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(event.truth_map.shape[0])[event.truth_map >= 0]
#     truth_map_to_edges = truth_map_to_edges[edge_mask]
# 
#     new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
#     new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(truth_map_to_edges.shape[0])[truth_map_to_edges >= 0]
#     event.truth_map = new_map.to(event.truth_map.device)
# 
# def handle_weighting(event, weighting_config):
#     """
#     Take the specification of the weighting and convert this into float values. The default is:
#     - True edges have weight 1.0
#     - Negative edges have weight 1.0
# 
#     The weighting_config can be used to change this behaviour. For example, we might up-weight target particles - that is edges that pass:
#     - y == 1
#     - primary == True
#     - pt > 1 GeV
#     - etc. As desired.
# 
#     We can also down-weight (i.e. mask) edges that are true, but not of interest. For example, we might mask:
#     - y == 1
#     - primary == False
#     - pt < 1 GeV
#     - etc. As desired.
#     """
# 
#     # Set the default values, which will be overwritten if specified in the config
#     weights = torch.zeros_like(event.y, dtype=torch.float)
#     weights[event.y == 0] = 1.0
# 
#     for weight_spec in weighting_config:
#         weight_val = weight_spec["weight"]
#         weights[get_weight_mask(event, weight_spec["conditions"])] = weight_val
# 
#     return weights
# 
# def get_weight_mask(event, weight_conditions):
# 
#     graph_mask = torch.ones_like(event.y)
# 
#     for condition_key, condition_val in weight_conditions.items():
#         assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
#         condition_lambda = get_condition_lambda(condition_key, condition_val)
#         value_mask = condition_lambda(event)
#         graph_mask = graph_mask * map_tensor_handler(value_mask, output_type="edge-like", num_nodes = event.num_nodes, edge_index = event.edge_index, truth_map = event.truth_map)
# 
#     return graph_mask
# 
# def map_tensor_handler(input_tensor: torch.Tensor, 
#                        output_type: str, 
#                        input_type: str = None, 
#                        truth_map: torch.Tensor = None, 
#                        edge_index: torch.Tensor = None,
#                        track_edges: torch.Tensor = None,
#                        num_nodes: int = None, 
#                        num_edges: int = None, 
#                        num_track_edges: int = None,
#                        aggr: str = None):
#     """
#     A general function to handle arbitrary maps of one tensor type to another. Types are "node-like", "edge-like" and "track-like".
#     - Node-like: The input tensor is of the same size as the number of nodes in the graph
#     - Edge-like: The input tensor is of the same size as the number of edges in the graph, that is, the *constructed* graph
#     - Track-like: The input tensor is of the same size as the number of true track edges in the event, that is, the *truth* graph
# 
#     To visualize:
#                     (n)
#                      ^
#                     / \ 
#       edge_to_node /   \ track_to_node
#                   /     \
#                  /       \
#                 /         \
#                /           \
#               /             \
# node_to_edge /               \ node_to_track
#             /                 \
#            |                   | 
#            v     edge_to_track v
#           (e) <-------------> (t)
#             track_to_edge
# 
#     Args:
#         input_tensor (torch.Tensor): The input tensor to be mapped
#         output_type (str): The type of the output tensor. One of "node-like", "edge-like" or "track-like"
#         input_type (str, optional): The type of the input tensor. One of "node-like", "edge-like" or "track-like". Defaults to None, and will try to infer the type from the input tensor, if num_nodes and/or num_edges are provided.
#         truth_map (torch.Tensor, optional): The truth map tensor. Defaults to None. Used for mappings to/from track-like tensors.
#         num_nodes (int, optional): The number of nodes in the graph. Defaults to None. Used for inferring the input type.
#         num_edges (int, optional): The number of edges in the graph. Defaults to None. Used for inferring the input type.
#         num_track_edges (int, optional): The number of track edges in the graph. Defaults to None. Used for inferring the input type.
#     """
# 
#     # Refactor the above switch case into a dictionary
#     mapping_dict = {
#         ("node-like", "edge-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_nodes_to_edges(input_tensor, edge_index, aggr),
#         ("edge-like", "node-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_edges_to_nodes(input_tensor, edge_index, aggr, num_nodes),
#         ("node-like", "track-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_nodes_to_tracks(input_tensor, track_edges, aggr),
#         ("track-like", "node-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_tracks_to_nodes(input_tensor, track_edges, aggr, num_nodes),
#         ("edge-like", "track-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_edges_to_tracks(input_tensor, truth_map),
#         ("track-like", "edge-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_tracks_to_edges(input_tensor, truth_map, num_edges),
#     }
# 
#     if num_track_edges is None and truth_map is not None:
#         num_track_edges = truth_map.shape[0]
#     if num_track_edges is None and track_edges is not None:
#         num_track_edges = track_edges.shape[1]
#     if num_edges is None and edge_index is not None:
#         num_edges = edge_index.shape[1]
#     if input_type is None:
#         input_type, input_tensor = infer_input_type(input_tensor, num_nodes, num_edges, num_track_edges)
# 
#     if input_type == output_type:
#         return input_tensor
#     elif (input_type, output_type) in mapping_dict:
#         return mapping_dict[(input_type, output_type)](input_tensor, truth_map, edge_index, track_edges, num_nodes, num_edges, num_track_edges, aggr)
#     else:
#         raise ValueError(f"Mapping from {input_type} to {output_type} not supported")
# 
# def infer_input_type(input_tensor: torch.Tensor, num_nodes: int = None, num_edges: int = None, num_track_edges: int = None):
#     """
#     Tries to infer the input type from the input tensor and the number of nodes, edges and track-edges in the graph.
#     If the input tensor cannot be matched to any of the provided types, it is assumed to be node-like.
#     """
# 
#     if num_nodes is not None and input_tensor.shape[0] == num_nodes:
#         return "node-like", input_tensor
#     elif num_edges is not None and num_edges in input_tensor.shape:
#         return "edge-like", input_tensor
#     elif num_track_edges is not None and num_track_edges in input_tensor.shape:
#         return "track-like", input_tensor
#     elif num_track_edges is not None and num_track_edges//2 in input_tensor.shape:
#         return "track-like", torch.cat([input_tensor, input_tensor], dim=0)
#     else:
#         return "node-like", input_tensor
# 
# def map_tracks_to_edges(tracklike_input: torch.Tensor, truth_map: torch.Tensor, num_edges: int = None):
#     """
#     Map an track-like tensor to a edge-like tensor. This is done by sending the track value through the truth map, where the truth map is >= 0. Note that where truth_map == -1,
#     the true edge has not been constructed in the edge_index. In that case, the value is set to NaN.
#     """
# 
#     if num_edges is None:
#         num_edges = int(truth_map.max().item() + 1)
#     edgelike_output = torch.zeros(num_edges, dtype=tracklike_input.dtype, device=tracklike_input.device)
#     edgelike_output[truth_map[truth_map >= 0]] = tracklike_input[truth_map >= 0]
#     edgelike_output[truth_map[truth_map == -1]] = float("nan")
#     return edgelike_output
