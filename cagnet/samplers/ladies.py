import torch
import torch_sparse
from collections import defaultdict

def start_time(timer):
    timer.record()

def stop_time(start_timer, stop_timer):
    stop_timer.record()
    torch.cuda.synchronize()
    return start_timer.elapsed_time(stop_timer)

def ladies_sampler(adj_matrix, batch_size, frontier_size, mb_count, n_layers, train_nodes):
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
    adj_matrices = [None] * n_layers
    node_count = adj_matrix.size(0)
    torch.cuda.nvtx.range_pop()
    print(f"instantiations: {stop_time(start_timer, stop_timer)}")

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-gen-minibatch-vtxs")
    torch.manual_seed(0)
    # Generate minibatch vertices
    for i in range(mb_count):
        idx = torch.randperm(train_nodes.size(0))[:batch_size]
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
            forntier_indices_cols = current_frontier.view(-1)
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
        p_num = torch.sparse_coo_tensor(indices=p_num_indices, values=p_num_values, size=(mb_count, node_count))
        p_den = torch.sparse.sum(p_num, dim=1)

        for j in range(mb_count):
            if p_den._nnz() != mb_count:
                print("ERROR nnz: {p_den._nnz()} mb_count: {mb_count}")
            p_den_mb = p_den._values()[j].item()
            p = torch.div(p_num, p_den)
        torch.cuda.nvtx.range_pop()
        print(f"compute-p: {stop_time(start_timer, stop_timer)}")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-pre-loop")
        # max_prob = torch.sparse.max(p, dim=1)
        p_torch_sparse = torch_sparse.tensor.SparseTensor.from_torch_sparse_coo_tensor(p)
        max_prob = torch_sparse.reduce.reduction(p_torch_sparse, dim=1, reduce="max")
        next_frontier = torch.cuda.LongTensor(mb_count, frontier_size).fill_(node_count)
        sampled_count = torch.cuda.IntTensor(mb_count).fill_(0)
        torch.cuda.nvtx.range_pop()
        print(f"pre-loop: {stop_time(start_timer, stop_timer)}")

        iter_count = 0
        torch.cuda.nvtx.range_push("nvtx-sampling")
        while (sampled_count < frontier_size).all():
            iter_count += 1
            start_time(sample_start_timer)
            torch.cuda.nvtx.range_push("nvtx-sampling-iter")
            # sample u.a.r. sampled_count vertices per minibatch from vertices in the minibatch's neighborhood
            # neighbored is vertices with nonzero probabilities
            # TODO: What if a vertex is sampled more than once? (idk if w/o replacement is possible)
            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-rand-vtxs")
            sampled_verts_ids = torch.cuda.LongTensor(mb_count, frontier_size) # indices into p._values()
            nnz_count = 0
            for j in range(mb_count):
                nnz_mb_mask = p._indices()[0, :] == j
                p_values_mb = p._values()[nnz_mb_mask]
                sampled_verts_ids[i] = torch.randint(p_values_mb.size(0), (frontier_size,))
                sampled_verts_ids[i] += nnz_count
                nnz_count += p_values_mb.size(0)
            torch.cuda.nvtx.range_pop()
            timing_dict["rand_vtxs"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-p-loc")
            p_loc = p._values()[sampled_verts_ids.view(-1)].view(mb_count, frontier_size)

            torch.cuda.nvtx.range_pop()
            timing_dict["p_loc"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-dart-throw")
            dart_throws = torch.cuda.FloatTensor(mb_count, frontier_size)
            for j in range(mb_count):
                dart_throws[i] = torch.cuda.FloatTensor(frontier_size).uniform_(to=max_prob[i])
            torch.cuda.nvtx.range_pop()
            timing_dict["dart_throw"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-filter-darts")
            dart_misses = p_loc <= dart_throws
            dart_hits = p_loc > dart_throws
            hit_verts_ids = sampled_verts_ids[dart_hits]
            sampled_verts = p._indices()[1][hit_verts_ids]
            torch.cuda.nvtx.range_pop()
            timing_dict["filter_darts"].append(stop_time(start_timer, stop_timer))
            
            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-set-probs")
            next_frontier[dart_hits] = sampled_verts
            p._values()[hit_verts_ids] = 0.0
            torch.cuda.nvtx.range_pop()
            timing_dict["set_probs"].append(stop_time(start_timer, stop_timer))

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-count-samples")
            sampled_count = torch.sum(next_frontier != node_count)
            torch.cuda.nvtx.range_pop()
            timing_dict["count_samples"].append(stop_time(start_timer, stop_timer))
            torch.cuda.nvtx.range_pop() # nvtx-sampling-iter
            timing_dict["sampling_iters"].append(stop_time(sample_start_timer, sample_stop_timer))
        torch.cuda.nvtx.range_pop() # nvtx-sampling

        # print(f"iter_count: {iter_count}")
        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-construct-nextf")
        next_frontier = torch.cat((next_frontier, batches), dim=1)
        torch.cuda.nvtx.range_pop()
        print(f"construct-nextf: {stop_time(start_timer, stop_timer)}")

        for j in range(mb_count):
            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-select-mtxs")
            if i == 0:
                row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz).cuda(), batches[j,:]))
            else:
                row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz).cuda(), \
                                                                            current_frontier[j, :]))
            row_select_mtx_values = torch.cuda.FloatTensor(nnz).fill_(1.0)

            col_select_mtx_indices = torch.stack((next_frontier[j], torch.arange(start=0, \
                                                        end=next_frontier[j].size(0)).cuda()))
            col_select_mtx_values = torch.cuda.FloatTensor(next_frontier[j].size(0)).fill_(1.0)
            torch.cuda.nvtx.range_pop()
            print(f"select-mtxs: {stop_time(start_timer, stop_timer)}")

            # multiply row_select matrix with adj_matrix
            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-row-select-spgemm")
            sampled_indices, sampled_values = torch_sparse.spspmm(row_select_mtx_indices.long(), 
                                                        row_select_mtx_values,
                                                        adj_matrix._indices(), adj_matrix._values(),
                                                        nnz, node_count, node_count, coalesced=True)
            torch.cuda.nvtx.range_pop()
            print(f"row-select-spgemm: {stop_time(start_timer, stop_timer)}")

            # multiply adj_matrix with col_select matrix
            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-col-select-spgemm")
            sampled_indices, sampled_values = torch_sparse.spspmm(sampled_indices, sampled_values,
                                                        col_select_mtx_indices.long(), col_select_mtx_values,
                                                        nnz, node_count, next_frontier[j].size(0), 
                                                        coalesced=True)
            torch.cuda.nvtx.range_pop()
            print(f"col-select-spgemm: {stop_time(start_timer, stop_timer)}")
            # layer_adj_matrix = adj_matrix[current_frontier[:,0], next_frontier]

            start_time(start_timer)
            torch.cuda.nvtx.range_push("nvtx-set-sample")
            current_frontier[j, :] = next_frontier[j, :]
            adj_matrix_sample = torch.sparse_coo_tensor(indices=sampled_indices, values=sampled_values, \
                                            size=(nnz, next_frontier[j].size(0)))
            adj_matrices[i] = adj_matrix_sample
            torch.cuda.nvtx.range_pop()
            print(f"set-sample: {stop_time(start_timer, stop_timer)}")
    
    print(f"total_time: {stop_time(total_start_timer, total_stop_timer)}")
    for k, v in timing_dict.items():
        print(f"{k} total_time: {sum(v)} avg_time {sum(v) / len(v)}")
    print(f"iter_count: {iter_count}")
    return batches[:, 0], next_frontier, adj_matrices
