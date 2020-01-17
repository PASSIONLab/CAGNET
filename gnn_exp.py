import os.path as osp

import torch
import torch.distributed
from torch.multiprocessing import Process

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


        
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.NLLLoss()
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

learning_rate = 0.01
# learning_rate = 1e-6
# for epoch in range(201):
for epoch in range(2):
    outputs = GCNFunc.apply(inputs, weight, adj_matrix)
    # logits = torch.argmax(outputs, dim=1)

    loss = criterion(outputs, data.y)
    loss.backward()

    # acc = outputs.eq(data.y).sum().item()
    # acc = acc / list(data.y.size())[0]

    accs = [] 
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        print(mask)
        pred = outputs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item()
        acc = acc / mask.sum().item()
        accs.append(acc)

    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, accs[0], accs[1], accs[2]))

    with torch.no_grad():
        weight -= learning_rate * weight.grad
        weight.grad.zero_()

