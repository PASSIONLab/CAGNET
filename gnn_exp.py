import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, MessagePassing  # noqa
# from torch_geometric.nn import ChebConv, MessagePassing  # noqa
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # No normalization just to make things easier
        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

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
        # x, adj_matrix, func = args[0], args[1], args[2]

        ctx.save_for_backward(inputs, weight, adj_matrix)

        agg_feats = torch.mm(adj_matrix, inputs)

        z = torch.mm(agg_feats, weight)

        # ctx.func = func
        return z
    
    @staticmethod
    def backward(ctx, grad_output):
        # is grad_output just G?
        # print("grad_output: " + str(grad_output.size()))
        inputs, weight, adj_matrix = ctx.saved_tensors

        grad_input = torch.mm(torch.mm(adj_matrix, grad_output), weight.t())
        grad_weight = torch.mm(torch.mm(inputs.t(), adj_matrix), grad_output)

        # return torch.cuda.FloatTensor(2708, 1433).fill_(0), torch.cuda.FloatTensor(1433, 7).fill_(0), None
        return grad_input, grad_weight, None


        
class Net(torch.nn.Module):
    def __init__(self, func):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.func = func
        # self.conv2 = GCNConv(16, dataset.num_classes)
        # self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        # self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, weight, adj_matrix):
        """
        x = F.relu(self.conv1(x, edge_index))
        # No dropout just to make things easier
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        return x
        """
        # return GCNFunc.apply(x, adj_matrix, self.func)
        return GCNFunc.apply(x, weight, adj_matrix)

criterion = torch.nn.CrossEntropyLoss()
model, data = Net(criterion).to(device), data.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
for name, param in model.named_parameters():
    if param.requires_grad:
            print(name, param.data, param.data.size())

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
# for epoch in range(4):
    outputs = GCNFunc.apply(inputs, weight, adj_matrix)
    final_outputs = torch.argmax(outputs, dim=1)

    # loss = (final_outputs - data.y).pow(2).sum()
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

# def train():
#     # model.train()
#     # outputs = model(inputs, weight, adj_matrix)
#     outputs = GCNFunc.apply(inputs, weight, adj_matrix)
#     print(weight)
#     optimizer.zero_grad()
#     # F.nll_loss(model(inputs, weight, adj_matrix)[data.train_mask], data.y[data.train_mask]).backward()
#     loss = criterion(outputs, data.y)
#     loss.backward()
#     optimizer.step()
#     print(weight)
# 
# 
# def test():
#     model.eval()
#     # logits, accs = model(), []
#     logits, accs = model(inputs, weight, adj_matrix), []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         pred = logits[mask].max(1)[1]
#         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#         accs.append(acc)
#     return accs
# 
# 
# best_val_acc = test_acc = 0
# # for epoch in range(1, 201):
# for epoch in range(1, 2):
#     train()
#     train_acc, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#     print(log.format(epoch, train_acc, best_val_acc, test_acc))
