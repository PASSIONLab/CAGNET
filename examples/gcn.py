import argparse
import math
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torch_geometric
from torch_geometric.datasets import Planetoid, PPI
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse
from gcn_conv import GCNConv
import socket

class GCN(nn.Module):
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, rank, size, group, device):
    super(GCN, self).__init__()
    self.layers = nn.ModuleList()
    self.rank = rank
    self.size = size
    self.group = group
    self.device = device
    self.timings = dict()

    # input layer
    self.layers.append(GCNConv(in_feats, n_hidden, self.device))
    # hidden layers
    for i in range(n_layers - 1):
        self.layers.append(GCNConv(n_hidden, n_hidden, self.device))
    # output layer
    self.layers.append(GCNConv(n_hidden, n_classes, self.device))

    # # input layer
    # self.mlp_layers.append(GCNConvMLP(in_feats, n_hidden, self.device))
    # # hidden layers
    # for i in range(n_layers - 1):
    #     self.mlp_layers.append(GCNConvMLP(n_hidden, n_hidden, self.device))
    # # output layer
    # self.mlp_layers.append(GCNConvMLP(n_hidden, n_classes, self.device))

  def forward(self, graph, inputs, ampbyp):
    h = inputs
    for l, layer in enumerate(self.layers):
      h = layer(self, graph, ampbyp, h)
      if l != len(self.layers) - 1:
        h = F.relu(h)

    h = F.log_softmax(h, dim=1)
    return h

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx, normalization):
    if not normalization:
        return adj_part

    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].double(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))

    return adj_part

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def oned_partition(rank, size, inputs, adj_matrix, data, features, classes, device, normalize):
    node_count = inputs.size(0)
    n_per_proc = math.ceil(float(node_count) / size)

    am_partitions = None
    am_pbyp = None

    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[rank + 1] - vtx_indices[rank]
        am_pbyp, _ = split_coo(am_partitions[rank], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            if i == size - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank], normalize)
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank], normalize)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i], normalize)

        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / size), dim=0)

        adj_matrix_loc = am_partitions[rank]
        inputs_loc = input_partitions[rank]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs.size: {inputs.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp

def evaluate(model, graph, features, labels, nid, nid_count, ampbyp, group):
    model.eval()
    with torch.no_grad():
        non_eval_timings = copy.deepcopy(model.timings)
        logits = model(graph, features, ampbyp, degrees)
        model.timings = non_eval_timings # don't include evaluation timings

        # # all-gather logits across ranks
        # logits_recv = []
        # for i in range(len(ampbyp)):
        #     logits_recv.append(torch.FloatTensor(ampbyp[0].size(1), logits.size(1)))

        # if logits.size(0) != ampbyp[0].size(1):
        #     pad_row = ampbyp[0].size(1) - logits.size(0)
        #     logits = torch.cat((logits, torch.FloatTensor(pad_row, logits.size(1))), dim=0)

        # dist.all_gather(logits_recv, logits, group)

        # padding = graph.size(0) - ampbyp[0].size(1) * (len(ampbyp) - 1)
        # logits_recv[-1] = logits_recv[-1][:padding,:]

        # logits = torch.cat(logits_recv)

        logits = logits[nid]
        labels = labels[nid]
        if logits.size(0) > 0:
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
        else:
            correct = torch.tensor(0)

        dist.all_reduce(correct, op=dist.reduce_op.SUM, group=group)
        return correct.item() * 1.0 / nid_count
        # return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    device = torch.device('cuda')
    graphname = "Cora"
    if graphname == "Cora":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', graphname)
        dataset = Planetoid(path, graphname, transform=T.NormalizeFeatures())
        data = dataset[0]
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        adj_matrix = edge_index

    # Initialize distributed environment with SLURM
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"
    
    dist.init_process_group(backend=args.dist_backend)
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}", flush=True)

    if args.normalize:
        adj_matrix, _ = add_remaining_self_loops(adj_matrix, num_nodes=inputs.size(0))

    group = dist.new_group(list(range(size)))
    features_loc, g_loc, ampbyp = oned_partition(rank, size, inputs, adj_matrix, data, \
                                                      inputs, num_classes, device, args.normalize)

    features_loc = features_loc.to(device)
    g_loc = g_loc.to(device)
    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)


    # create GCN model
    torch.manual_seed(0)
    model = GCN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.n_layers,
                      rank,
                      size,
                      group,
                      device)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_per_proc = math.ceil(float(inputs.size(0)) / size)

    rank_train_mask = torch.split(data.train_mask, n_per_proc, dim=0)[rank]
    rank_test_mask = torch.split(data.test_mask, n_per_proc, dim=0)[rank]
    labels_rank = torch.split(data.y, n_per_proc, dim=0)[rank]
    rank_train_nids = rank_train_mask.nonzero().squeeze()
    rank_test_nids = rank_test_mask.nonzero().squeeze()

    train_nid = data.train_mask.nonzero().squeeze()
    test_nid = data.test_mask.nonzero().squeeze()

    torch.manual_seed(0)
    total_start = time.time()
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}", flush=True)
        model.train()

        # forward
        logits = model(g_loc, features_loc, ampbyp)
        loss = F.cross_entropy(logits[rank_train_nids], labels_rank[rank_train_nids], reduction="sum") 

        loss_recv = []
        for i in range(size // args.replication):
            loss_recv.append(torch.cuda.FloatTensor(loss.size()))
        dist.all_gather(loss_recv, loss, group)
        loss_recv[rank] = loss
        loss = sum(loss_recv) / train_nid.size(0)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        """
        # acc = evaluate(model, g_loc, features_loc, labels, val_nid, ampbyp, ampbyp_dgl, degrees, col_groups[0])
        # acc = evaluate(model, g_loc, features_loc, labels, val_nid, \
        acc = evaluate(model, g_loc, features_loc, labels_rank, rank_val_nids, \
                            val_mask.nonzero().squeeze().size(0), ampbyp, ampbyp_dgl, degrees, col_groups[0])
        print("Rank: {:05d} | Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(rank, epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000), flush=True)
        """

    """
    # print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    barrier_start = time.time()
    dist.barrier()
    model.timings["total"] += time.time() - total_start
    model.timings["barrier"] += time.time() - barrier_start
    print()
    # # acc = evaluate(model, g_loc, features_loc, labels, test_nid, ampbyp, ampbyp_dgl, degrees, col_groups[0])
    # acc = evaluate(model, g_loc, features_loc, labels, test_nid, \
    acc = evaluate(model, g_loc, features_loc, labels_rank, rank_test_nids, \
                        test_mask.nonzero().squeeze().size(0), ampbyp, ampbyp_dgl, degrees, col_groups[0])
    print("Test Accuracy {:.4f}".format(acc))

    print(flush=True)
    dist.barrier()
    print("Timings")
    model.timings["comp"] = model.timings["scomp"] + model.timings["dcomp"]
    model.timings["comm"] = model.timings["bcast"] + model.timings["reduce"] + model.timings["op"]
    # print("rank, total, scomp, dcomp, bcast, reduce, op, barrier")
    # print(f"{rank}, {model.timings['total']}, {model.timings['scomp']}, {model.timings['dcomp']}, {model.timings['bcast']}, {model.timings['reduce']}, {model.timings['op']}, {model.timings['barrier']}")
    print(f"rank: {rank} timings: {model.timings}")
    """
    print(f"rank: {rank} {logits}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--replication", type=int, default=1,
                        help="replciation factor for 1.5D algorithm")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                            help='distributed backend')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument('--normalize', action="store_true",
                            help='normalize adjacency matrix')
    args = parser.parse_args()
    print(args)

    main(args)
