import os
import os.path as osp
import argparse

import torch
import torch.distributed as dist

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, degree, to_dense_adj
import torch_geometric.transforms as T

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_add

def norm(edge_index, num_nodes, edge_weight=None, improved=False,
         dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def block_row(adj_matrix, inputs, weight, rank, size):
    n_per_proc = int(adj_matrix.size(1) / size)
    am_partitions = list(torch.split(adj_matrix, n_per_proc, dim=1))

    z_loc = torch.zeros(n_per_proc, inputs.size(1))
    
    inputs_recv = torch.zeros(inputs.size())

    for i in range(size):
        part_id = (rank + i) % size

        z_loc += torch.mm(am_partitions[part_id], inputs) 

        src = (rank + 1) % size
        dst = rank - 1
        if dst < 0:
            dst = size - 1

        if size == 1:
            continue

        if rank == 0:
            dist.send(tensor=inputs, dst=dst)
            dist.recv(tensor=inputs_recv, src=src)
        else:
            dist.recv(tensor=inputs_recv, src=src)
            dist.send(tensor=inputs, dst=dst)
        
        inputs = inputs_recv.clone()

    z_loc = torch.mm(z_loc, weight)
    return z_loc

def outer_product(adj_matrix, grad_output, rank, size, group):
    n_per_proc = adj_matrix.size(1)
    
    # A * G^l
    ag = torch.mm(adj_matrix, grad_output)

    # reduction on A * G^l low-rank matrices
    dist.all_reduce(ag, op=dist.reduce_op.SUM, group=group)

    # partition A * G^l by block rows and get block row for this process
    # TODO: this might not be space-efficient
    red_partitions = list(torch.split(ag, n_per_proc, dim=0))
    grad_input = red_partitions[rank]

    return grad_input

def outer_product2(inputs, ag, rank, size, group):
    # (H^(l-1))^T * (A * G^l)
    grad_weight = torch.mm(inputs, ag)

    # reduction on grad_weight low-rank matrices
    dist.all_reduce(grad_weight, op=dist.reduce_op.SUM, group=group)

    return grad_weight

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, rank, size, group):
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.rank = rank
        ctx.size = size
        ctx.group = group

        # agg_feats = torch.mm(adj_matrix.t(), inputs)
        # z = torch.mm(agg_feats, weight)

        z = block_row(adj_matrix.t(), inputs, weight, rank, size)

        return z

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors
        rank = ctx.rank
        size = ctx.size
        group = ctx.group

        # grad_input = torch.mm(torch.mm(adj_matrix, grad_output), weight.t())
        # First backprop equation
        ag = outer_product(adj_matrix, grad_output, rank, size, group)
        grad_input = torch.mm(ag, weight.t())


        # Second backprop equation (reuses the A * G^l computation)
        # grad_weight = torch.mm(torch.mm(inputs.t(), adj_matrix), grad_output)
        grad_weight = outer_product2(inputs.t(), ag, rank, size, group)

        return grad_input, grad_weight, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, optimizer, data, rank, size, group):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, rank, size, group)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, rank, size, group)

    output_parts = [torch.zeros(outputs.size())] * size
    output_parts = []
    for i in range(size):
        output_parts.append(torch.zeros(outputs.size()))

    dist.all_gather(output_parts, outputs)

    output_parts[rank] = outputs
    outputs = torch.cat(output_parts, dim=0)

    optimizer.zero_grad()
    loss = F.nll_loss(outputs[data.train_mask], data.y[data.train_mask])
    loss.backward()
    
    optimizer.step()

    return outputs

def test(outputs, data):
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def run(rank, size, inputs, adj_matrix, data, features, classes, device):
    best_val_acc = test_acc = 0
    outputs = None
    group = dist.new_group(list(range(size)))

    adj_matrix_loc = torch.rand(adj_matrix.size(0), int(adj_matrix.size(1) / size))
    inputs_loc = torch.rand(int(inputs.size(0) / size), inputs.size(1))

    torch.manual_seed(0)
    weight1_nonleaf = torch.rand(features, 16, requires_grad=True)
    weight1_nonleaf = weight1_nonleaf.to(device)
    weight1_nonleaf.retain_grad()

    weight2_nonleaf = torch.rand(16, classes, requires_grad=True)
    weight2_nonleaf = weight2_nonleaf.to(device)
    weight2_nonleaf.retain_grad()

    weight1 = Parameter(weight1_nonleaf)
    weight2 = Parameter(weight2_nonleaf)

    optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)

    am_partitions = None
    # Scatter partitions to the different processes
    # It probably makes more sense to read the partitions as inputs but will change later.
    if rank == 0:
        # TODO: Maybe I do want grad here. Unsure.
        am_partitions = torch.split(adj_matrix, int(adj_matrix.size(0) / size), dim=1)
        input_partitions = torch.split(inputs, int(inputs.size(0) / size), dim=0)
        am_partitions = list(am_partitions)
        input_partitions = list(input_partitions)

        for i in range(size):
            am_partitions[i] = am_partitions[i].contiguous()
            input_partitions[i] = input_partitions[i].contiguous()

        # for i in range(size):
            # input_partitions[i].requires_grad = True

        dist.scatter(adj_matrix_loc, src=0, scatter_list=am_partitions, group=group)
        dist.scatter(inputs_loc, src=0, scatter_list=input_partitions, group=group)
    else:
        dist.scatter(adj_matrix_loc, src=0, group=group)
        dist.scatter(inputs_loc, src=0, group=group)

    for epoch in range(1, 201):
    # for epoch in range(1):
        # outputs = train(inputs, weight1, weight2, adj_matrix, optimizer, data, rank, size)
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, optimizer, data, rank, size, group)
        train_acc, val_acc, tmp_test_acc = test(outputs, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print(outputs)

def init_process(rank, size, inputs, adj_matrix, data, features, classes, device, fn, 
                            backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, inputs, adj_matrix, data, features, classes, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--processes', metavar='P', type=int,
                        help='Number of processes')
    args = parser.parse_args()
    print(args)
    P = args.processes
    if P is None:
        P = 1
    
    print("Processes: " + str(P))

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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')
    device = torch.device('cpu')

    mp.set_start_method('spawn')
    # model, data = Net().to(device), data.to(device)
    data = data.to(device)
    data.x.requires_grad = True
    inputs = data.x.to(device)
    inputs.requires_grad = True
    data.y = data.y.to(device)

    edge_index = data.edge_index
    adj_matrix = to_dense_adj(edge_index)[0].to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    processes = []
    for rank in range(P):
        p = Process(target=init_process, args=(rank, P, inputs, adj_matrix, 
                        data, dataset.num_features, dataset.num_classes, device, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
