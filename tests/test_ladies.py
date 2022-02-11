import argparse
import examples.gcn_15d
import LADIES.pytorch_ladies

import numpy as np
import torch

def main(args):
    """
    # Testing sampled neighbors, probability distribution, and normalized Laplacian 
    # with original LADIES code

    mine, mine_mtx, mine_batch = examples.gcn_15d.main(args)
    mine = mine.cuda()
    them, them_mtx, them_batch = LADIES.pytorch_ladies.main(args)
    them = torch.from_numpy(them).cuda()

    mine_sampled_neighbors = mine._indices()[1,:].squeeze()
    them_sampled_neighbors = np.nonzero(them).squeeze()

    mine_sampled_probs = mine._values().float()
    them_sampled_probs = them[them_sampled_neighbors].float()

    if not torch.equal(mine_sampled_neighbors, them_sampled_neighbors):
        print("indices don't match")
        print(f"mine_sampled_neighbors.size: {mine_sampled_neighbors.size()}")
        print(f"mine_sampled_neighbors: {mine_sampled_neighbors}")
        print(f"them_sampled_neighbors.size: {them_sampled_neighbors.size()}")
        print(f"them_sampled_neighbors: {them_sampled_neighbors}")
    else:
        print("indices match")

    if not torch.allclose(mine_sampled_probs, them_sampled_probs, atol=0.001, rtol=0.0):
        print("probabilities don't match")
        print(f"mine_sampled_probs.size: {mine_sampled_probs.size()}")
        print(f"mine_sampled_probs: {mine_sampled_probs}")
        print(f"them_sampled_probs.size: {them_sampled_probs.size()}")
        print(f"them_sampled_probs: {them_sampled_probs}")
    else:
        print("probabilities match")

    mine_mtx_normalization = mine_mtx._values()
    them_mtx_normalization = torch.from_numpy(them_mtx.data).float().cuda()

    if not torch.allclose(mine_mtx_normalization, them_mtx_normalization, atol=0.001, rtol=0.0):
        print("laplacians don't match")
        unmatch_idx = ~torch.isclose(mine_mtx_normalization, them_mtx_normalization, atol=0.001, rtol=0.0)
        print(f"unmatch_idx: {unmatch_idx.nonzero()}")
        print(f"mine_mtx_normalization: {mine_mtx_normalization[unmatch_idx]}")
        print(f"them_mtx_normalization: {them_mtx_normalization[unmatch_idx]}")
    else:
        print("laplacians match")

    them_batch = torch.from_numpy(them_batch).cuda()

    if not torch.equal(mine_batch, them_batch):
        print("batches don't match")
        print(f"mine_batch: {mine_batch}")
        print(f"them_batch: {them_batch}")
    else:
        print("batches match")
    """


    """
    All vertices in new frontier exist in aggregated neighborhood of old frontier
    """
    
    # Compute aggregated neighborhood
    current_frontier_all, next_frontier_all, adj_matrices_all, g_loc = examples.gcn_15d.main(args)
    torch.set_printoptions(edgeitems=25)
    print(f"g_loc: {g_loc}")

    print("")
    print("Beginning LADIES sampler test")
    for i in range(args.n_bulkmb):
        print(f"Testing minibatch {i}")
        current_frontier = current_frontier_all[i] # current_frontier for minibatch i
        next_frontier = next_frontier_all[i] # next_frontier for minibatch i
        adj_matrices = adj_matrices_all[i] # adj_matrices for minibatch i

        agg_neighbors = []
        for v in current_frontier:
            vtx = v.item()
            neighbors_mask = g_loc._indices()[0,:] == vtx
            neighbors = (g_loc._indices()[1,:])[neighbors_mask]
            agg_neighbors += neighbors.tolist()

        agg_neighbors = torch.IntTensor(agg_neighbors)
        agg_neighbors = agg_neighbors.unique()

        # Verify each vertex in next_frontier exists in aggregated neighorhood
        vertices_match = True
        for v in next_frontier:
            vtx = v.item()
            if vtx not in agg_neighbors:
                vertices_match = False
                print(f"vtx {vtx} not in aggregated neighborhood")

        if vertices_match:
            print("next_frontier within aggregated neighorhood")

        """
        All edges between the two frontiers of vertices exist in the sample
        """
        adj_matrix = adj_matrices[0] # assume only one layer for now
        nnz_count = 0
        edges_match = True
        for u in current_frontier:
            for v in next_frontier:
                u_vtx = u.item()
                v_vtx = v.item()

                neighbors_mask = g_loc._indices()[0,:] == u_vtx
                neighbors = (g_loc._indices()[1,:])[neighbors_mask]
                neighbors_nnz = (g_loc._values())[neighbors_mask]
                
                if v_vtx in neighbors:
                    v_vtx_idx = (neighbors == v_vtx).nonzero().item()
                    if adj_matrix._values()[nnz_count] != neighbors_nnz[v_vtx_idx]:
                        print(f"nnz differs u: {u_vtx} v: {v_vtx} sample_nnz: {adj_matrix._values()[nnz_count]} g_loc_nnz: {neighbors_nnz[v_vtx_idx]}")
                        edges_match = False
                    nnz_count += 1

        if edges_match:
            print("all edges between two frontiers exist in sampled adj matrix")

        """
        No other edges exist in sample (i.e. edges with an endpoint outside the two frontiers of vertices)
        """
        if nnz_count != adj_matrix._nnz():
            print("there exist other edges in sampled adj matrix")
        else:
            print("no other edges exist in sampled adj matrix")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test LADIES sampling code')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="number of vertices in minibatch")
    parser.add_argument("--samp-num", type=int, default=64,
                        help="number of vertices per layer of layer-wise minibatch")
    parser.add_argument('--normalize', action="store_true",
                            help='normalize adjacency matrix')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--replication', default=1, type=int,
                            help='partitioning strategy to use')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Avaiable GPU ID')
    parser.add_argument('--sample-method', type=str, default='ladies',
                        help='Sampled Algorithms: ladies/fastgcn/full')
    parser.add_argument('--batch-num', type=int, default= 10,
                        help='Maximum Batch Number')
    parser.add_argument('--pool-num', type=int, default= 10,
                        help='Number of Pool')
    parser.add_argument('--n-bulkmb', type=int, default=1,
                        help='Number of minibatches to sample in bulk')
    args = parser.parse_args()
    print(args)

    main(args)
