import torch
import torch_sparse
from collections import defaultdict

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu

def start_time(timer):
    timer.record()

def stop_time(start_timer, stop_timer):
    stop_timer.record()
    torch.cuda.synchronize()
    return start_timer.elapsed_time(stop_timer)

def ladies_sampler(adj_matrix, batch_size, frontier_size, mb_count, n_layers, n_darts, train_nodes):
    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    sample_start_timer = torch.cuda.Event(enable_timing=True)
    sample_stop_timer = torch.cuda.Event(enable_timing=True)

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-instantiations")
    # frontiers = torch.cuda.IntTensor(n_layers, batch_size, mb_count)
    batches = torch.cuda.IntTensor(mb_count, batch_size) # initially the minibatch, note row-major
    current_frontier = torch.cuda.IntTensor(mb_count, batch_size + frontier_size)
    # frontiers = torch.cuda.IntTensor(n_layers - 1, batch_size + frontier_size, mb_count)
    adj_matrices = [[None] * n_layers for x in range(mb_count)] # adj_matrices[i][j] --  mb i layer j
    node_count = adj_matrix.size(0)
    torch.cuda.nvtx.range_pop()
    print(f"instantiations: {stop_time(start_timer, stop_timer)}")

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-gen-minibatch-vtxs")

    torch.manual_seed(0)
    vertex_perm = torch.randperm(train_nodes.size(0))
    # Generate minibatch vertices
    for i in range(mb_count):
        idx = vertex_perm[(i * batch_size):((i + 1) * batch_size)]
        batches[i,:] = train_nodes[idx]
    torch.cuda.nvtx.range_pop()
    print(f"get-minibatch-vtxs: {stop_time(start_timer, stop_timer)}")

    start_time(total_start_timer)
    for i in range(n_layers):
        if i == 0:
            nnz = batches[0, :].size(0)
        else:
            nnz = current_frontier[0, :].size(0)

        # A * Q^i
        # indices would change based on mb_count
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-gen-sparse-frontier")
        if i == 0:
            frontier_indices_rows = torch.arange(mb_count).cuda()
            frontier_indices_rows = frontier_indices_rows.repeat_interleave(nnz)
            frontier_indices_cols = batches.view(-1)
            frontier_indices = torch.stack((frontier_indices_rows, frontier_indices_cols))
        else:
            frontier_indices_rows = torch.arange(mb_count).cuda()
            frontier_indices_rows = frontier_indices_rows.repeat_interleave(nnz)
            frontier_indices_cols = current_frontier.view(-1)
            frontier_indices = torch.stack((frontier_indices_rows, frontier_indices_cols))
        frontier_values = torch.cuda.FloatTensor(nnz * mb_count).fill_(1.0)
        adj_mat_squared = torch.pow(adj_matrix._values(), 2)
        torch.cuda.nvtx.range_pop()
        print(f"gen-sparse-frontier: {stop_time(start_timer, stop_timer)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-probability-spgemm")
        p_num_indices, p_num_values = torch_sparse.spspmm(frontier_indices.long(), frontier_values,
                                        adj_matrix._indices().long(), adj_mat_squared,
                                        mb_count, node_count, node_count, coalesced=True)
        torch.cuda.nvtx.range_pop()
        print(f"probability-spgemm: {stop_time(start_timer, stop_timer)}")

        print(f"p_num_indices.size(): {p_num_indices.size()}")
        print(f"p_num_values.size(): {p_num_values.size()}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-compute-p")
        p_num = torch.sparse_coo_tensor(indices=p_num_indices, values=p_num_values, \
                                                size=(mb_count, node_count))
        p_den = torch.sparse.sum(p_num, dim=1)

        for j in range(mb_count):
            if p_den._nnz() != mb_count:
                print("ERROR nnz: {p_den._nnz()} mb_count: {mb_count}")
            p_den_mb = p_den._values()[j].item()
            p = torch.div(p_num, p_den_mb)
        torch.cuda.nvtx.range_pop()
        print(f"compute-p: {stop_time(start_timer, stop_timer)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-pre-loop")
        next_frontier = torch.sparse_coo_tensor(indices=p._indices(),
                                                    values=torch.cuda.LongTensor(p._nnz()).fill_(0),
                                                    size=(mb_count, node_count))
        sampled_count = torch.cuda.IntTensor(mb_count).fill_(0)

        neighbor_sizes = torch.sparse_coo_tensor(indices=p._indices(),
                                                    values=torch.cuda.LongTensor(p._nnz()).fill_(1),
                                                    size=(mb_count, node_count))
        neighbor_sizes = torch.sparse.sum(neighbor_sizes, dim=1)

        psum_neighbor_sizes = torch.cumsum(neighbor_sizes._values(), dim=0).roll(1)
        psum_neighbor_sizes[0] = 0

        torch.cuda.nvtx.range_pop()
        print(f"pre-loop: {stop_time(start_timer, stop_timer)}")

        iter_count = 0
        torch.cuda.nvtx.range_push("nvtx-sampling")
        while (sampled_count < frontier_size).any():
            iter_count += 1
            start_time(sample_start_timer)
            torch.cuda.nvtx.range_push("nvtx-sampling-iter")

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-comp-maxprob")
            p_torch_sparse = torch_sparse.tensor.SparseTensor.from_torch_sparse_coo_tensor(p)
            max_prob = torch_sparse.reduce.reduction(p_torch_sparse, dim=1, reduce="max")
            torch.cuda.nvtx.range_pop()
            timing_dict["max-prob"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-gen-darts")
            dartx_values = torch.cuda.FloatTensor(n_darts * mb_count).uniform_()
            darty_values = torch.cuda.FloatTensor(n_darts * mb_count).uniform_()
            torch.cuda.nvtx.range_pop()
            timing_dict["gen-darts"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-dart-throw")
            compute_darts_gpu(dartx_values, darty_values, neighbor_sizes._values(), psum_neighbor_sizes,
                                    max_prob, n_darts, mb_count)

            torch.cuda.nvtx.range_pop()
            timing_dict["dart-throw"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-filter-darts")
            dart_hits_mask = torch.cuda.LongTensor(p._nnz()).fill_(0)
            throw_darts_gpu(dartx_values, darty_values, p._values(), dart_hits_mask, n_darts, mb_count)
            torch.cuda.nvtx.range_pop()
            timing_dict["filter_darts"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-add-to-frontier")
            next_frontier_values = torch.logical_or(dart_hits_mask, next_frontier._values()).long()
            next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                                                            values=next_frontier_values,
                                                            size=(mb_count, node_count))
            torch.cuda.nvtx.range_pop()
            timing_dict["add-to-frontier"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-count-samples")
            sampled_count = torch.sparse.sum(next_frontier_tmp, dim=1)._values()
            torch.cuda.nvtx.range_pop()
            timing_dict["count_samples"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-downsample")
            overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
            downsample_gpu(dart_hits_mask, next_frontier_tmp._indices()[0,:], overflow, next_frontier_tmp._nnz())
            torch.cuda.nvtx.range_pop()
            timing_dict["downsample"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-add-to-frontier2")
            next_frontier_values = torch.logical_or(dart_hits_mask, next_frontier._values()).long()
            next_frontier = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                                                        values=next_frontier_values,
                                                        size=(mb_count, node_count))
            torch.cuda.nvtx.range_pop()
            timing_dict["add-to-frontier2"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-count-samples2")
            sampled_count = torch.sparse.sum(next_frontier, dim=1)._values()
            torch.cuda.nvtx.range_pop()
            timing_dict["count_samples2"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-set-probs")
            dart_hits_mask = dart_hits_mask.bool()
            p._values()[dart_hits_mask] = 0.0
            torch.cuda.nvtx.range_pop()
            timing_dict["set-probs"].append(stop_time(start_timer, stop_timer))

            torch.cuda.nvtx.range_pop() # nvtx-sampling-iter
            timing_dict["sampling_iters"].append(stop_time(sample_start_timer, sample_stop_timer))

        torch.cuda.nvtx.range_pop() # nvtx-sampling

        # TODO: Might need to downsample if > frontier_size vertices were sampled for a minibatch
        print(f"next_frontier.frontier-sizes: {torch.sparse.sum(next_frontier, dim=1)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-construct-nextf")
        next_frontier = torch.masked_select(next_frontier._indices()[1,:], \
                                                next_frontier._values().bool()).view(mb_count, frontier_size)
        next_frontier = torch.cat((next_frontier, batches), dim=1)
        torch.cuda.nvtx.range_pop()
        print(f"construct-nextf: {stop_time(start_timer, stop_timer)}")

        torch.cuda.nvtx.range_push("nvtx-select-rowcols")
        # 1. Make the row/col select matrices
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-select-mtxs")

        if i == 0:
            row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz * mb_count).cuda(),
                                                  batches.view(-1)))
        else:
            # TODO i > 0 (make sure size args to spspmm are right)
            print("ERROR i > 0")

        row_select_mtx_values = torch.cuda.FloatTensor(row_select_mtx_indices[0].size(0)).fill_(1.0)

        repeated_next_frontier = next_frontier.repeat_interleave(nnz, dim=0)
        scale_mtx = torch.arange(start=0, end=node_count * nnz * mb_count, step=node_count).cuda()
        repeated_next_frontier.add_(scale_mtx[:, None])
        col_select_mtx_rows=repeated_next_frontier.view(-1)
        col_select_mtx_cols = torch.arange(next_frontier.size(1)).cuda().repeat(nnz * mb_count,1).view(-1)
        col_select_mtx_indices = torch.stack((col_select_mtx_rows, col_select_mtx_cols))
        col_select_mtx_values = torch.cuda.FloatTensor(col_select_mtx_rows.size(0)).fill_(1.0)
        torch.cuda.nvtx.range_pop()
        timing_dict["select-mtxs"].append(stop_time(start_timer, stop_timer))

        # 2. Multiply row_select matrix with adj_matrix
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-row-select-spgemm")
        sampled_indices, sampled_values = torch_sparse.spspmm(row_select_mtx_indices.long(),
                                                    row_select_mtx_values,
                                                    adj_matrix._indices(), adj_matrix._values(),
                                                    nnz * mb_count, node_count, node_count, coalesced=True)
        torch.cuda.nvtx.range_pop()
        timing_dict["row-select-spgemm"].append(stop_time(start_timer, stop_timer))

        # 3. Expand sampled rows
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-row-select-expand")
        sampled_indices[1,:].add_(sampled_indices[0,:], alpha=node_count)
        torch.cuda.nvtx.range_pop()
        timing_dict["row-select-expand"].append(stop_time(start_timer, stop_timer))

        # 4. Multiply sampled rows with col_select matrix
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-col-select-spgemm")
        sampled_indices, sampled_values = torch_sparse.spspmm(sampled_indices, sampled_values,
                                                    col_select_mtx_indices.long(), col_select_mtx_values,
                                                    nnz * mb_count, node_count * nnz * mb_count, next_frontier.size(1),
                                                    coalesced=True)
        torch.cuda.nvtx.range_pop()
        timing_dict["col-select-spgemm"].append(stop_time(start_timer, stop_timer))

        # 5. Store sampled matrices
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-set-sample")
        adj_matrix_sample = torch.sparse_coo_tensor(indices=sampled_indices, values=sampled_values, \
                                                    size=(nnz * mb_count, next_frontier.size(1)))
        for mb in range(mb_count):
            mb_sampled_rows = torch.arange(start=batch_size * mb, end=batch_size * (mb + 1)).cuda()
            adj_matrices[mb][i] = torch.index_select(adj_matrix_sample, 0, mb_sampled_rows)

        current_frontier = next_frontier
        torch.cuda.nvtx.range_pop()
        timing_dict["set-sample"].append(stop_time(start_timer, stop_timer))

        torch.cuda.nvtx.range_pop()


    print(f"total_time: {stop_time(total_start_timer, total_stop_timer)}")
    for k, v in timing_dict.items():
        print(f"{k} total_time: {sum(v)} avg_time {sum(v) / len(v)}")
    print(f"iter_count: {iter_count}")
    return batches, next_frontier, adj_matrices
