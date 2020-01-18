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

KipfWelling = False

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

in_channels = dataset.num_features
out_channels = dataset.num_classes
seed = 0

print("in: " + str(in_channels) + " out: " + str(out_channels))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True, bias=False)
        torch.manual_seed(seed)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True, bias=False)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

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

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

data.x.requires_grad = True
inputs = data.x.to(device)

torch.manual_seed(seed)
weight1 = torch.rand(in_channels, 16, requires_grad=True)
weight1 = weight1.to(device)
weight1.retain_grad()

torch.manual_seed(seed)
weight2 = torch.rand(16, out_channels, requires_grad=True)
weight2 = weight2.to(device)
weight2.retain_grad()

edge_index = data.edge_index
edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))
adj_matrix = to_dense_adj(edge_index)[0].to(device)

print("adj_matrix size: " + str(adj_matrix.size()))

learning_rate = 1e-1
# learning_rate = 1e-6
# for epoch in range(201):
for epoch in range(1):
    if KipfWelling:
        model.train()
        model.zero_grad()
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()

        for W in model.parameters():
            print(W.grad.data.size())
            W.data -= learning_rate * W.grad.data

        model.eval()
        logits, accs = model(), []
        # accs = [] 
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            # pred = outputs[mask].max(1)[1]
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item()
            acc = acc / mask.sum().item()
            accs.append(acc)

        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, accs[0], accs[1], accs[2]))
    else:
        tmp_out = GCNFunc.apply(inputs, weight1, adj_matrix)
        # tmp_out = F.dropout(tmp_out)
        outputs = GCNFunc.apply(tmp_out, weight2, adj_matrix)

        # loss = criterion(outputs, data.y)
        # loss.backward()
        F.nll_loss(outputs[data.train_mask], data.y[data.train_mask]).backward()

        with torch.no_grad():
            print(weight1.grad.size())
            print(weight2.grad.size())
            weight1 -= learning_rate * weight1.grad
            weight1.grad.zero_()

            weight2 -= learning_rate * weight2.grad
            weight2.grad.zero_()

        accs = [] 
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = outputs[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item()
            acc = acc / mask.sum().item()
            accs.append(acc)

        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, accs[0], accs[1], accs[2]))

