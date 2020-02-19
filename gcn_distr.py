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

from torch.multiprocessing import Manager, Process

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

    z_loc = torch.cuda.FloatTensor(n_per_proc, inputs.size(1)).fill_(0)
    
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

def broad_func(adj_matrix, inputs, rank, size, group):
    n_per_proc = int(adj_matrix.size(1) / size)
    am_partitions = list(torch.split(adj_matrix, n_per_proc, dim=1))

    z_loc = torch.cuda.FloatTensor(n_per_proc, inputs.size(1)).fill_(0)
    # z_loc = torch.zeros(n_per_proc, inputs.size(1))
    
    inputs_recv = torch.cuda.FloatTensor(inputs.size())
    # inputs_recv = torch.zeros(inputs.size())

    for i in range(size):
        if i == rank:
            inputs_recv = inputs.clone()

        dist.broadcast(inputs_recv, src=i, group=group)

        z_loc += torch.mm(am_partitions[i], inputs_recv) 

    return z_loc

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, rank, size, group, func):
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.rank = rank
        ctx.size = size
        ctx.group = group

        ctx.func = func

        # z = block_row(adj_matrix.t(), inputs, weight, rank, size)
        z = broad_func(adj_matrix.t(), inputs, rank, size, group)
        z = torch.mm(z, weight)

        z.requires_grad = True
        ctx.z = z

        if func is F.log_softmax:
            h = func(z, dim=1)
        elif func is F.relu:
            h = func(z)
        else:
            h = z

        return h

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors
        rank = ctx.rank
        size = ctx.size
        group = ctx.group

        func = ctx.func
        z = ctx.z

        with torch.set_grad_enabled(True):
            if func is F.log_softmax:
                func_eval = func(z, dim=1)
            elif func is F.relu:
                func_eval = func(z)
            else:
                func_eval = z

            sigmap = torch.autograd.grad(outputs=func_eval, inputs=z, grad_outputs=grad_output)[0]
            grad_output = sigmap

        # First backprop equation
        ag = outer_product(adj_matrix, grad_output, rank, size, group)
        grad_input = torch.mm(ag, weight.t())


        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = outer_product2(inputs.t(), ag, rank, size, group)

        return grad_input, grad_weight, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, optimizer, data, rank, size, group):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, rank, size, group, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, rank, size, group, F.log_softmax)

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask.bool(), outputs.size(0), dim=0)[rank]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[rank]

    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
        loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        loss.backward()
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size()).fill_(0)).sum()
        # fake_loss = (outputs * torch.zeros(outputs.size())).sum()
        fake_loss.backward()

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

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        am_partitions = torch.split(adj_matrix, int(adj_matrix.size(0) / size), dim=1)
        input_partitions = torch.split(inputs, int(inputs.size(0) / size), dim=0)

        adj_matrix_loc = am_partitions[rank]
        inputs_loc = input_partitions[rank]

    for epoch in range(1, 201):
    # for epoch in range(1):
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, optimizer, data, rank, size, group)
        print("Epoch: {:03d}".format(epoch))

    # All-gather outputs to test accuracy
    output_parts = [torch.zeros(outputs.size())] * size
    output_parts = []
    for i in range(size):
        output_parts.append(torch.cuda.FloatTensor(outputs.size()).fill_(0))

    dist.all_gather(output_parts, outputs)
    outputs = torch.cat(output_parts, dim=0)

    train_acc, val_acc, tmp_test_acc = test(outputs, data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

    print(log.format(200, train_acc, best_val_acc, test_acc))
    print(outputs)
    return outputs

def init_process(rank, size, inputs, adj_matrix, data, features, classes, device, outputs, fn, 
                            backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    run_outputs = fn(rank, size, inputs, adj_matrix, data, features, classes, device)
    if outputs is not None:
        outputs[rank] = run_outputs.detach()

def main(P, correctness_check):
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    seed = 0

    mp.set_start_method('spawn', force=True)
    # device = torch.device('cpu')
    device = torch.device('cuda')

    data = data.to(device)
    data.x.requires_grad = True
    inputs = data.x.to(device)
    inputs.requires_grad = True
    data.y = data.y.to(device)

    edge_index = data.edge_index
    adj_matrix = to_dense_adj(edge_index)[0].to(device)

    processes = []
    outputs = None
    for rank in range(P):
        if correctness_check and rank == 0:
            manager = Manager()
            outputs = manager.dict()

        p = Process(target=init_process, args=(rank, P, inputs, adj_matrix, 
                        data, dataset.num_features, dataset.num_classes, device, outputs, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if outputs is not None:
        return outputs[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--processes', metavar='P', type=int,
                        help='Number of processes')
    parser.add_argument('--correctness', metavar='C', type=str,
                        help='Run correctness check')
    args = parser.parse_args()
    print(args)
    P = args.processes
    correctness_check = args.correctness
    if P is None:
        P = 1

    if correctness_check is None or correctness_check == "nocheck":
        correctness_check = False
    else:
        correctness_check = True
    
    print("Processes: " + str(P))
    print("Correctness: " + str(correctness_check))
    print(main(P, correctness_check))
