import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, MessagePassing  # noqa
# from torch_geometric.nn import ChebConv, MessagePassing  # noqa
from torch_geometric.utils import add_self_loops, degree

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

print(type(data.x))

class GCNFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, *args):
        x, edge_index, func = args[0], args[1], args[2]

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        lin = lin.to(device)
        
        x = lin(x)

        ctx.func = func
        ctx.save_for_backward(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors
        print(z)
        print(z[0].size())
        
        # return z[0], z[0], None
        return torch.cuda.FloatTensor(2708, 1433).fill_(0), z[0], None

        
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

    def forward(self, x):
        edge_index = data.edge_index
        """
        x = F.relu(self.conv1(x, edge_index))
        # No dropout just to make things easier
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        return x
        """
        x = x.to(device)
        edge_index = edge_index.to(device)
        return GCNFunc.apply(x, edge_index, self.func)

criterion = torch.nn.CrossEntropyLoss()
model, data = Net(criterion).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(data.y.size())
print(dataset.num_classes)
print(data.y)
print(torch.max(data.y))

data.x.requires_grad = True

def train():
    model.train()
    optimizer.zero_grad()
    outputs = model(data.x)
    print(outputs.size())
    loss = criterion(outputs, data.y)
    loss.backward()
    # F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    # logits, accs = model(), []
    logits, accs = model(data.x), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
# for epoch in range(1, 201):
for epoch in range(1, 2):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
