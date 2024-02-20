import os.path as osp
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', "cora")
data = Planetoid(path, "cora")[0]

print(f"data: {data}", flush=True)
loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    edge_label_index=data.edge_index,
)

sampled_data = next(iter(loader))
print(f"sampled_data: {sampled_data}", flush=True)
