import os.path as osp
import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

seed = 0

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix):
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

torch.manual_seed(seed)
weight1 = torch.rand(dataset.num_features, 16, requires_grad=True)
weight1 = weight1.to(device)
weight1.retain_grad()

weight2 = torch.rand(16, dataset.num_classes, requires_grad=True)
weight2 = weight2.to(device)
weight2.retain_grad()

# model, data = Net().to(device), data.to(device)
data = data.to(device)
data.x.requires_grad = True
inputs = data.x.to(device)

edge_index = data.edge_index
edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))
adj_matrix = to_dense_adj(edge_index)[0].to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam([Parameter(weight1), Parameter(weight2)], lr=0.01)

learning_rate = 1e-1

def train():
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix)
    optimizer.zero_grad()
    F.nll_loss(outputs[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


    return outputs

def test(outputs):
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    outputs = train()
    train_acc, val_acc, tmp_test_acc = test(outputs)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

