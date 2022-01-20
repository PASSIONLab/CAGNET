import torch
import torch_sparse

def ladies_sampler(adj_matrix, batch_size, frontier_size, mb_count, n_layers, train_nodes):
    frontiers = torch.cuda.IntTensor(n_layers, batch_size, mb_count)
    node_count = adj_matrix.size(0)

    torch.manual_seed(0)
    for i in range(mb_count):
        idx = torch.randperm(train_nodes.size(0))[:batch_size]
        frontiers[0,:,i] = train_nodes[idx]

    for i in range(1, n_layers + 1):
        nnz = frontiers[i - 1, :, 0].size(0)

        # A * Q^i
        # indices would change based on mb_count
        frontier_indices = torch.stack((torch.cuda.IntTensor(batch_size).fill_(0), frontiers[i - 1,:,0]))
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

        print(f"p: {p}")
        print(f"p.sum: {torch.sparse.sum(p)}")

        print(f"neighbors: {p._indices()[1,:]}")

        max_prob = torch.max(p._values())
        frontier_next = torch.cuda.LongTensor(frontier_size).fill_(node_count)
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
            print(f"sampled_verts: {sampled_verts}")
            
            frontier_next[dart_hits] = sampled_verts
            p._values()[hit_verts_ids] = 0.0

            sampled_count += hit_count

            print(f"sampled_count: {sampled_count}")
            print(f"hit_count: {hit_count}")

        print(f"frontier_next: {frontier_next}")
        return p
