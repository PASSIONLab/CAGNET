import math
import torch
import torch.distributed as dist
import torch_sparse
from collections import defaultdict

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, compute_darts_select_gpu, \
                                    throw_darts_select_gpu, compute_darts1d_gpu, throw_darts1d_gpu, \
                                    normalize_gpu, shift_rowselect_gpu, shift_colselect_gpu

def start_time(timer):
    timer.record()

def stop_time(start_timer, stop_timer):
    stop_timer.record()
    torch.cuda.synchronize()
    return start_timer.elapsed_time(stop_timer)

def dist_spgemm1D(mata, matb, rank, size, group):
    chunk_size = math.ceil(float(mata.size(1) / size))
    matc = torch.sparse_coo_tensor(size=(mata.size(0), matb.size(1))).cuda()

    for i in range(size):
        matb_recv_nnz = torch.cuda.IntTensor([matb._nnz()])
        dist.broadcast(matb_recv_nnz, src=i, group=group)

        if i == rank:
            matb_recv_indices = matb._indices().clone()
            matb_recv_values = matb._values().clone()
        else:
            matb_recv_indices = torch.cuda.LongTensor(2, matb_recv_nnz.item())
            matb_recv_values = torch.cuda.FloatTensor(matb_recv_nnz.item())

        dist.broadcast(matb_recv_indices, src=i, group=group)
        dist.broadcast(matb_recv_values, src=i, group=group)

        chunk_col_start = i * chunk_size
        chunk_col_stop = min((i + 1) * chunk_size, mata.size(1))
        chunk_col_size = chunk_col_stop - chunk_col_start
        chunk_col_mask = (mata._indices()[1, :] >= chunk_col_start) & (mata._indices()[1, :] < chunk_col_stop)

        mata_chunk_indices = mata._indices()[:, chunk_col_mask]
        mata_chunk_indices[1,:] -= chunk_col_start
        mata_chunk_values = mata._values()[chunk_col_mask]

        matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_chunk_indices, \
                                                    mata_chunk_values, matb_recv_indices, \
                                                    matb_recv_values, mata.size(0), \
                                                    chunk_col_size, matb.size(1), coalesced=True)

        matc_chunk = torch.sparse_coo_tensor(matc_chunk_indices, matc_chunk_values, size=matc.size())
        matc += matc_chunk

    return matc._indices(), matc._values()

def ladies_sampler(adj_matrix, batches, batch_size, frontier_size, mb_count_total, n_layers, n_darts, \
                        rank, size, group):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    sample_start_timer = torch.cuda.Event(enable_timing=True)
    sample_stop_timer = torch.cuda.Event(enable_timing=True)

    select_start_timer = torch.cuda.Event(enable_timing=True)
    select_stop_timer = torch.cuda.Event(enable_timing=True)

    select_iter_start_timer = torch.cuda.Event(enable_timing=True)
    select_iter_stop_timer = torch.cuda.Event(enable_timing=True)

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    adj_matrices = [[None] * n_layers for x in range(mb_count_total)] # adj_matrices[i][j] --  mb i layer j
    current_frontier = torch.cuda.IntTensor(mb_count_total, batch_size + frontier_size)

    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    mb_count = batches.size(0)

    start_time(total_start_timer)
    for i in range(n_layers):
        if i == 0:
            nnz = batch_size
        else:
            nnz = current_frontier[0, :].size(0)

        # A * Q^i
        # indices would change based on mb_count
        
        # TODO: assume n_layers=1 for now
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-probability-spgemm")
        print(f"batches: {batches}")
        print(f"adj_matrices: {adj_matrix}")
        p_num_indices, p_num_values = dist_spgemm1D(batches, adj_matrix, rank, size, group)
        print(f"p_num_indices: {p_num_indices}")
        print(f"p_num_values: {p_num_values}")
        torch.cuda.nvtx.range_pop()
        print(f"probability-spgemm: {stop_time(start_timer, stop_timer)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-compute-p")
        p = torch.sparse_coo_tensor(indices=p_num_indices, 
                                        values=p_num_values, 
                                        size=(mb_count, node_count_total))
        p_den = torch.cuda.FloatTensor(mb_count).fill_(0)
        p_den = p_den.scatter_add_(0, p._indices()[0, :], p._values())
        normalize_gpu(p._values(), p_den, p._indices()[0, :], p._nnz())
        print(f"compute-p: {stop_time(start_timer, stop_timer)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-pre-loop")
        next_frontier = torch.sparse_coo_tensor(indices=p._indices(),
                                                    values=torch.cuda.LongTensor(p._nnz()).fill_(0),
                                                    size=(mb_count, node_count_total))
        sampled_count = torch.cuda.IntTensor(mb_count).fill_(0)

        torch.cuda.nvtx.range_pop()
        print(f"pre-loop: {stop_time(start_timer, stop_timer)}")

        iter_count = 0
        selection_iter_count = 0
        torch.cuda.nvtx.range_push("nvtx-sampling")
        p_rowsum = torch.cuda.FloatTensor(mb_count).fill_(0)
        
        # underfull_minibatches = (sampled_count < frontier_size).any()
        underfull_minibatches = True
        while underfull_minibatches:
            iter_count += 1
            start_time(sample_start_timer)
            torch.cuda.nvtx.range_push("nvtx-sampling-iter")

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-prob-rowsum")
            p_rowsum.fill_(0)
            p_rowsum.scatter_add_(0, p._indices()[0, :], p._values())
            ps_p_rowsum = torch.cumsum(p_rowsum, dim=0).roll(1)
            ps_p_rowsum[0] = 0
            ps_p_values = torch.cumsum(p._values(), dim=0).roll(1)
            ps_p_values[0] = 0
            torch.cuda.nvtx.range_pop()
            timing_dict["prob-rowsum"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-gen-darts")
            dart_values = torch.cuda.FloatTensor(n_darts * mb_count).uniform_()
            torch.cuda.nvtx.range_pop()
            timing_dict["gen-darts"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-dart-throw")
            compute_darts1d_gpu(dart_values, p_rowsum, ps_p_rowsum, n_darts, mb_count)
            torch.cuda.nvtx.range_pop()
            timing_dict["dart-throw"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-filter-darts")
            dart_hits_count = torch.cuda.LongTensor(p._nnz()).fill_(0)
            throw_darts1d_gpu(dart_values, ps_p_values, dart_hits_count, n_darts * mb_count, p._nnz())
            torch.cuda.nvtx.range_pop()
            timing_dict["filter_darts"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-add-to-frontier")
            next_frontier_values = torch.logical_or(dart_hits_count, next_frontier._values()).int()
            next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                                                            values=next_frontier_values,
                                                            size=(mb_count, node_count_total))
            torch.cuda.nvtx.range_pop()
            timing_dict["add-to-frontier"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-count-samples")
            # sampled_count = torch.sparse.sum(next_frontier_tmp, dim=1)._values()
            sampled_count = sampled_count.fill_(0)
            sampled_count.scatter_add_(0, next_frontier_tmp._indices()[0, :], next_frontier_values)
            torch.cuda.nvtx.range_pop()
            timing_dict["count_samples"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-select-preproc")
            overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
            overflowed_minibatches = (overflow > 0).any()
            timing_dict["select-preproc"].append(stop_time(start_timer, stop_timer))

            start_time(select_start_timer)
            torch.cuda.nvtx.range_push("nvtx-dart-selection")

            while overflowed_minibatches:
                start_time(select_iter_start_timer)
                torch.cuda.nvtx.range_push("nvtx-selection-iter")

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-psoverflow")
                selection_iter_count += 1
                ps_overflow = torch.cumsum(overflow, dim=0)
                total_overflow = ps_overflow[-1].item()
                timing_dict["select-psoverflow"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-reciprocal")
                dart_hits_inv = dart_hits_count.reciprocal()
                dart_hits_inv[dart_hits_inv == float("inf")] = 0
                timing_dict["select-reciprocal"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-instmtx")
                dart_hits_inv_mtx = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                                                                values=dart_hits_inv,
                                                                size=(mb_count, node_count_total))
                timing_dict["select-instmtx"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-invsum")
                # dart_hits_inv_sum = torch.sparse.sum(dart_hits_inv_mtx, dim=1)._values()
                dart_hits_inv_sum = torch.cuda.FloatTensor(mb_count).fill_(0)
                dart_hits_inv_sum.scatter_add_(0, next_frontier._indices()[0, :], dart_hits_inv)
                timing_dict["select-invsum"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-psinv")
                ps_dart_hits_inv_sum = torch.cumsum(dart_hits_inv_sum, dim=0).roll(1)
                ps_dart_hits_inv_sum[0] = 0
                timing_dict["select-psinv"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-computedarts")
                dart_select = torch.cuda.FloatTensor(total_overflow).uniform_()
                # Compute darts for selection 
                compute_darts_select_gpu(dart_select, dart_hits_inv_sum, ps_dart_hits_inv_sum, ps_overflow,
                                                mb_count, total_overflow)
                timing_dict["select-computedarts"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-throwdarts")
                # Throw selection darts 
                ps_dart_hits_inv = torch.cumsum(dart_hits_inv, dim=0)
                throw_darts_select_gpu(dart_select, ps_dart_hits_inv, dart_hits_count, total_overflow,
                                            next_frontier._nnz())
                timing_dict["select-throwdarts"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-add-to-frontier")
                next_frontier_values = torch.logical_or(dart_hits_count, next_frontier._values()).int()
                next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                                                                values=next_frontier_values,
                                                                size=(mb_count, node_count_total))
                timing_dict["select-add-to-frontier"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-samplecount")
                # sampled_count = torch.sparse.sum(next_frontier_tmp, dim=1)._values()
                sampled_count = torch.cuda.IntTensor(mb_count).fill_(0)
                sampled_count.scatter_add_(0, next_frontier_tmp._indices()[0,:], next_frontier_tmp._values())
                timing_dict["select-samplecount"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-overflow")
                overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
                overflowed_minibatches = (overflow > 0).any()
                timing_dict["select-overflow"].append(stop_time(start_timer, stop_timer))

                timing_dict["select-iter"].append(stop_time(select_iter_start_timer, select_iter_stop_timer))

            torch.cuda.nvtx.range_pop() # nvtx-dart-selection
            timing_dict["dart-selection"].append(stop_time(select_start_timer, select_stop_timer))

            next_frontier_values = torch.logical_or(dart_hits_count, next_frontier._values()).long()
            next_frontier = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                                                            values=next_frontier_values,
                                                            size=(mb_count, node_count_total))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-set-probs")
            dart_hits_mask = dart_hits_count > 0
            p._values()[dart_hits_mask] = 0.0

            filled_minibatches = sampled_count == frontier_size
            filled_minibatches_mask = torch.gather(filled_minibatches, 0, p._indices()[0,:])
            p._values()[filled_minibatches_mask] = 0
            torch.cuda.nvtx.range_pop()
            timing_dict["set-probs"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-compute-bool")
            underfull_minibatches = (sampled_count < frontier_size).any()
            torch.cuda.nvtx.range_pop()
            timing_dict["compute-bool"].append(stop_time(start_timer, stop_timer))

            torch.cuda.nvtx.range_pop() # nvtx-sampling-iter
            timing_dict["sampling_iters"].append(stop_time(sample_start_timer, sample_stop_timer))

        torch.cuda.nvtx.range_pop() # nvtx-sampling

        # TODO: Might need to downsample if > frontier_size vertices were sampled for a minibatch
        print(f"next_frontier.frontier-sizes: {torch.sparse.sum(next_frontier, dim=1)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-construct-nextf")
        next_frontier_select = torch.masked_select(next_frontier._indices()[1,:], \
                                                next_frontier._values().bool()).view(mb_count, frontier_size)
        batches_select = torch.masked_select(batches._indices()[1,:], \
                                                batches._values().bool()).view(mb_count, batch_size)
        next_frontier_select = torch.cat((next_frontier_select, batches_select), dim=1)

        print(f"rank: {rank} next_frontier_select.size: {next_frontier_select.size()}")
        print(f"rank: {rank} batches_select.size: {batches_select.size()}")
        print(f"rank: {rank} next_frontier_select: {next_frontier_select}")
        print(f"rank: {rank} batches_select: {batches_select}")

        torch.cuda.nvtx.range_pop()
        print(f"construct-nextf: {stop_time(start_timer, stop_timer)}")
        
        bulk_row_select_rows = torch.arange(0, batches._nnz(), device=torch.device("cuda"))
        bulk_row_select_cols = batches._indices()[1,:]
        bulk_row_select_indices = torch.stack((bulk_row_select_rows, bulk_row_select_cols))
        bulk_row_select_values = torch.cuda.FloatTensor(batches._nnz()).fill_(1.0)
        
        sampled_indices, sampled_values = torch_sparse.spspmm(bulk_row_select_indices, bulk_row_select_values,
                                                                adj_matrix._indices(), adj_matrix._values(),
                                                                batch_size * mb_count, node_count_total, 
                                                                node_count_total, coalesced=True)

        row_shift = torch.cuda.LongTensor(sampled_values.size(0)).fill_(0)
        shift_rowselect_gpu(row_shift, sampled_indices[0,:], sampled_values.size(0), batch_size, node_count_total)
        sampled_indices[1,:] += row_shift

        col_shift = torch.cuda.LongTensor(next_frontier_select.numel()).fill_(0)
        bulk_col_select_rows = next_frontier_select.clone().reshape(-1)
        shift_colselect_gpu(col_shift, next_frontier_select.numel(), frontier_size + batch_size, node_count_total)
        bulk_col_select_rows += col_shift
        bulk_col_select_cols = torch.arange(0, next_frontier_select.numel(), device=torch.device("cuda"))
        bulk_col_select_cols = torch.remainder(bulk_col_select_cols, frontier_size + batch_size)
        bulk_col_select_indices = torch.stack((bulk_col_select_rows, bulk_col_select_cols))
        bulk_col_select_values = torch.cuda.FloatTensor(next_frontier_select.numel()).fill_(1.0)

        sampled_indices, sampled_values = torch_sparse.spspmm(sampled_indices, sampled_values,
                                                    bulk_col_select_indices, bulk_col_select_values,
                                                    batch_size * mb_count, node_count_total * mb_count, 
                                                    frontier_size + batch_size, coalesced=True)

        for j in range(mb_count):
            row_select_min = j * batch_size
            row_select_max = (j + 1) * batch_size
            sample_select_mask = (row_select_min <= sampled_indices[0,:]) & \
                                 (sampled_indices[0,:] < row_select_max)
            adj_matrix_sample_indices = sampled_indices[:, sample_select_mask]
            adj_matrix_sample_indices[0,:] -= row_select_min
            adj_matrix_sample_values = sampled_values[sample_select_mask]

            adj_matrix_sample = torch.sparse_coo_tensor(adj_matrix_sample_indices, adj_matrix_sample_values, \
                                            size=(batch_size, frontier_size + batch_size))
            adj_matrices[j][i] = adj_matrix_sample

        # torch.cuda.nvtx.range_push("nvtx-select-rowcols")
        # for j in range(mb_count):
        #     start_time(start_timer)
        #     torch.cuda.nvtx.range_push("nvtx-select-mtxs")
        #     if i == 0:
        #         row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz).cuda(), batches_select[j,:]))
        #     else:
        #         row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz).cuda(), \
        #                                                                     current_frontier[j, :]))
        #     row_select_mtx_values = torch.cuda.FloatTensor(nnz).fill_(1.0)

        #     col_select_mtx_indices = torch.stack((next_frontier_select[j], torch.arange(start=0, \
        #                                                 end=next_frontier_select[j].size(0)).cuda()))
        #     col_select_mtx_values = torch.cuda.FloatTensor(next_frontier_select[j].size(0)).fill_(1.0)
        #     torch.cuda.nvtx.range_pop()
        #     timing_dict["select-mtxs"].append(stop_time(start_timer, stop_timer))

        #     # multiply row_select matrix with adj_matrix
        #     start_time(start_timer)
        #     torch.cuda.nvtx.range_push("nvtx-row-select-spgemm")
        #     sampled_indices, sampled_values = torch_sparse.spspmm(row_select_mtx_indices.long(), 
        #                                                 row_select_mtx_values,
        #                                                 adj_matrix._indices(), adj_matrix._values(),
        #                                                 nnz, node_count, node_count, coalesced=True)
        #     torch.cuda.nvtx.range_pop()
        #     timing_dict["row-select-spgemm"].append(stop_time(start_timer, stop_timer))

        #     # multiply adj_matrix with col_select matrix
        #     start_time(start_timer)
        #     torch.cuda.nvtx.range_push("nvtx-col-select-spgemm")
        #     sampled_indices, sampled_values = torch_sparse.spspmm(sampled_indices, sampled_values,
        #                                                 col_select_mtx_indices.long(), col_select_mtx_values,
        #                                                 nnz, node_count, next_frontier_select[j].size(0), 
        #                                                 coalesced=True)
        #     torch.cuda.nvtx.range_pop()
        #     timing_dict["col-select-spgemm"].append(stop_time(start_timer, stop_timer))
        #     # layer_adj_matrix = adj_matrix[current_frontier[:,0], next_frontier]

        #     start_time(start_timer)
        #     torch.cuda.nvtx.range_push("nvtx-set-sample")
        #     current_frontier[j, :] = next_frontier_select[j, :]
        #     adj_matrix_sample = torch.sparse_coo_tensor(indices=sampled_indices, values=sampled_values, \
        #                                     size=(nnz, next_frontier_select[j].size(0)))
        #     adj_matrices[j][i] = adj_matrix_sample
        #     torch.cuda.nvtx.range_pop()
        #     timing_dict["set-sample"].append(stop_time(start_timer, stop_timer))
        # torch.cuda.nvtx.range_pop()
    
    print(f"total_time: {stop_time(total_start_timer, total_stop_timer)}")
    for k, v in timing_dict.items():
        print(f"{k} total_time: {sum(v)} avg_time {sum(v) / len(v)}")
    print(f"iter_count: {iter_count}")
    print(f"selection_iter_count: {selection_iter_count}")
    return batches_select, next_frontier_select, adj_matrices
