# From https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

import torch

from torch_geometric.data import Data

x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)
print Data(edge_index=[2, 5], x=[4, 2], y=[4])
