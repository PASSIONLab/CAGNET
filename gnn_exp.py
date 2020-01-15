import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, MessagePassing  # noqa
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

in_channels = dataset.num_features
out_channels = dataset.num_classes

print("in: " + str(in_channels) + " out: " + str(out_channels))

class GCNFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix):
        # inputs: H
        # adj_matrix: A
        # weight: W

        ctx.save_for_backward(inputs, weight, adj_matrix)

        agg_feats = torch.mm(adj_matrix, inputs)

        z = torch.mm(agg_feats, weight)

        return z
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors

        grad_input = torch.mm(torch.mm(adj_matrix, grad_output), weight.t())
        grad_weight = torch.mm(torch.mm(inputs.t(), adj_matrix), grad_output)

        return grad_input, grad_weight, None


        
criterion = torch.nn.CrossEntropyLoss()
data = data.to(device)

data.x.requires_grad = True
inputs = data.x.to(device)

weight = torch.rand(in_channels, out_channels, requires_grad=True)
weight = weight.to(device)
weight.retain_grad()

edge_index = data.edge_index
edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))
adj_matrix = to_dense_adj(edge_index)[0].to(device)

print("adj_matrix size: " + str(adj_matrix.size()))
print(adj_matrix)

# learning_rate = 1e-6
learning_rate = 1e4
for epoch in range(500):
    outputs = GCNFunc.apply(inputs, weight, adj_matrix)
    final_outputs = torch.argmax(outputs, dim=1)

    loss = criterion(outputs, data.y)
    loss.backward()
    print("Epoch: " + str(epoch) + " Loss: " + str(loss))
    final_outputs = torch.argmax(outputs, dim=1)

    diff = final_outputs - data.y
    diff = torch.abs(diff)
    diff_count = torch.sum(diff)
    print("diff_count: " + str(diff_count))

    with torch.no_grad():
        weight -= learning_rate * weight.grad
        weight.grad.zero_()

