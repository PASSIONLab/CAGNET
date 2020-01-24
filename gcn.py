import os.path as osp

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv  # noqa
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, 16, normalize=False, bias=False)
        self.conv2 = SAGEConv(16, dataset.num_classes, normalize=False, bias=False)
        self.conv1.aggr="add"
        self.conv2.aggr="add"
        with torch.no_grad():
            self.conv1.weight = Parameter(weight1.clone())
            self.conv2.weight = Parameter(weight2.clone())
        # self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        # self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))
        # x = F.relu(self.conv1(x, edge_index))
        x = self.conv1(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=0)
        # return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

torch.manual_seed(0)
weight1 = torch.rand(dataset.num_features, 16)
weight1 = weight1.to(device)

weight2 = torch.rand(16, dataset.num_classes)
weight2 = weight2.to(device)

model, data = Net().to(device), data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4)

def train():
    model.train()
    # optimizer.zero_grad()
    model.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    print(model())
    # optimizer.step()
    for W in model.parameters():
        W.data -= 0.1 * W.grad.data

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item()
        acc = acc / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
# for epoch in range(1, 201):
for epoch in range(1):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
