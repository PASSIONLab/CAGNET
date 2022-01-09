import os.path as osp
import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI, Reddit
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv # noqa

import time

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# dataset = PPI(path, 'train', T.NormalizeFeatures())
# dataset = Reddit(path, T.NormalizeFeatures())
# dataset = Yelp(path, T.NormalizeFeatures())
data = dataset[0]

seed = 0

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True, normalize=False, bias=False)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True, normalize=False, bias=False)

        self.conv1.node_dim = 0
        self.conv2.node_dim = 0

        with torch.no_grad():
            self.conv1.weight = Parameter(weight1)
            self.conv2.weight = Parameter(weight2)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(seed)
weight1 = torch.rand(dataset.num_features, 16)
weight1 = weight1.to(device)

weight2 = torch.rand(16, dataset.num_classes)
weight2 = weight2.to(device)

data.y = data.y.type(torch.LongTensor)
model, data = Net().to(device), data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    outputs = model()
    
    # Note: bool type removes warnings, unsure of perf penalty
    F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()]).backward()
    # F.nll_loss(outputs, torch.max(data.y, 1)[1]).backward()

    for W in model.parameters():
        if W.grad is not None:
            print(W.grad)

    optimizer.step()
    return outputs

def test(outputs):
    model.eval()
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(): 
    best_val_acc = test_acc = 0
    outputs = None

    tstart = time.time()

    # for epoch in range(1, 101):
    for epoch in range(1):
        outputs = train()
        train_acc, val_acc, tmp_test_acc = test(outputs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    tstop = time.time()
    print("Time: " + str(tstop - tstart))

    return outputs

if __name__=='__main__':
    print(main())
