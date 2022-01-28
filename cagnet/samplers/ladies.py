import torch
import torch_sparse

def ladies_sampler(adj_matrix, batch_size, frontier_size, mb_count, n_layers, train_nodes):
    # frontiers = torch.cuda.IntTensor(n_layers, batch_size, mb_count)
    batches = torch.cuda.IntTensor(batch_size, mb_count) # initially the minibatch
    current_frontier = torch.cuda.IntTensor(batch_size + frontier_size, mb_count)
    # frontiers = torch.cuda.IntTensor(n_layers - 1, batch_size + frontier_size, mb_count)
    adj_matrices = [None] * n_layers
    node_count = adj_matrix.size(0)

    torch.manual_seed(0)
    # Generate minibatch vertices
    for i in range(mb_count):
        idx = torch.randperm(train_nodes.size(0))[:batch_size]
        batches[:,i] = train_nodes[idx]

    for i in range(n_layers):
        if i == 0:
            nnz = batches[:, 0].size(0) # assumes mb_count == 1
        else:
            nnz = current_frontier[:, 0].size(0) # assumes mb_count == 1

        # A * Q^i
        # indices would change based on mb_count
        if i == 0:
            frontier_indices = torch.stack((torch.cuda.IntTensor(nnz).fill_(0), batches[:,0]))
        else:
            frontier_indices = torch.stack((torch.cuda.IntTensor(nnz).fill_(0), current_frontier[:,0]))
        frontier_values = torch.cuda.FloatTensor(nnz).fill_(1.0)
        adj_mat_squared = torch.pow(adj_matrix._values(), 2)
        p_num_indices, p_num_values = torch_sparse.spspmm(frontier_indices.long(), frontier_values, 
                                        adj_matrix._indices().long(), adj_mat_squared,
                                        mb_count, node_count, node_count, coalesced=True)

        p_num = torch.sparse_coo_tensor(indices=p_num_indices, values=p_num_values, size=(mb_count, node_count))
        p_den = torch.sparse.sum(p_num, dim=1)

        # Assumes mb_count = 1
        p_den = p_den._values().item()
        p = torch.div(p_num, p_den)

        max_prob = torch.max(p._values())
        next_frontier = torch.cuda.LongTensor(frontier_size).fill_(node_count)
        sampled_count = 0
        
        while sampled_count < frontier_size:
            # sample u.a.r. sampled_count vertices per minibatch from vertices in the minibatch's neighborhood
            # neighbored is vertices with nonzero probabilities
            # TODO: What if a vertex is sampled more than once (idk if w/o replacement is possible)
            sampled_verts_ids = torch.randint(p._nnz(), (frontier_size,))

            p_loc = p._values()[sampled_verts_ids]

            dart_throws = torch.cuda.FloatTensor(frontier_size).uniform_(to=max_prob)

            dart_misses = p_loc <= dart_throws
            dart_hits = p_loc > dart_throws
            hit_count = dart_hits.sum()
            hit_verts_ids = sampled_verts_ids[dart_hits]
            sampled_verts = p._indices()[1][hit_verts_ids]
            
            next_frontier[dart_hits] = sampled_verts
            p._values()[hit_verts_ids] = 0.0

            sampled_count = torch.sum(next_frontier != node_count)

        next_frontier = torch.cat((next_frontier, batches[:,0]), dim=0)

        if i == 0:
            row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz).cuda(), batches[:,0]))
        else:
            row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz).cuda(), current_frontier[:,0]))
        row_select_mtx_values = torch.cuda.FloatTensor(nnz).fill_(1.0)

        col_select_mtx_indices = torch.stack((next_frontier, torch.arange(start=0, end=next_frontier.size(0))\
                                                .cuda()))
        col_select_mtx_values = torch.cuda.FloatTensor(next_frontier.size(0)).fill_(1.0)

        # multiply row_select matrix with adj_matrix
        sampled_indices, sampled_values = torch_sparse.spspmm(row_select_mtx_indices.long(), 
                                                    row_select_mtx_values,
                                                    adj_matrix._indices(), adj_matrix._values(),
                                                    nnz, node_count, node_count, coalesced=True)
        # multiply adj_matrix with col_select matrix
        sampled_indices, sampled_values = torch_sparse.spspmm(sampled_indices, sampled_values,
                                                    col_select_mtx_indices.long(), col_select_mtx_values,
                                                    nnz, node_count, next_frontier.size(0), coalesced=True)
        # layer_adj_matrix = adj_matrix[current_frontier[:,0], next_frontier]

        current_frontier[:,0] = next_frontier
        adj_matrix_sample = torch.sparse_coo_tensor(indices=sampled_indices, values=sampled_values, \
                                        size=(nnz, next_frontier.size(0)))
        adj_matrices[i] = adj_matrix_sample

    return batches[:, 0], next_frontier, adj_matrices
