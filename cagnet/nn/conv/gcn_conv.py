import math
import torch
import torch.distributed as dist
import torch.nn as nn

from sparse_coo_tensor_cpp import spmm_gpu

def broad_func(self, graph, ampbyp, inputs):
    n_per_proc = math.ceil(float(graph.size(0) / self.size))

    z_loc = torch.cuda.FloatTensor(ampbyp[0].size(0), inputs.size(1), device=self.device).fill_(0)
    
    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=self.device).fill_(0)

    for i in range(self.size):
        if i == self.rank:
            inputs_recv = inputs.clone()
        elif i == self.size - 1:
            inputs_recv = torch.cuda.FloatTensor(ampbyp[i].size(1), \
                                                        inputs.size(1), \
                                                        device=self.device).fill_(0)

        dist.broadcast(inputs_recv, src=i, group=self.group)

        spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(), 
                        ampbyp[i].values(), ampbyp[i].size(0), 
                        ampbyp[i].size(1), inputs_recv, z_loc)

    return z_loc

def outer_product(mata, matb, group):
    matc = torch.mm(mata, matb)
    dist.all_reduce(matc, op=dist.reduce_op.SUM, group=group)

    return matc

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, device):
        super(GCNConv, self).__init__()

        weight_nonleaf = torch.rand(in_feats, out_feats, requires_grad=True)
        weight_nonleaf = weight_nonleaf.to(device)
        weight_nonleaf.retain_grad()
        self.weight = nn.Parameter(weight_nonleaf)

    def forward(self, gcn, graph, ampbyp, inputs):
        return GCNFunc.apply(gcn, graph, ampbyp, inputs, self.weight)
        

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, graph, ampbyp, inputs, weight):
        # inputs: H
        # graph: A
        # weight; W

        z = broad_func(self, graph, ampbyp, inputs)
        z = z.mm(weight)

        ctx.save_for_backward(inputs, weight)
        ctx.ampbyp = ampbyp
        ctx.graph = graph
        ctx.self = self

        return z

    @staticmethod
    def backward(ctx, grad_output):
        graph = ctx.graph
        ampbyp = ctx.ampbyp
        inputs, weight = ctx.saved_tensors
        self = ctx.self

        # Assumes graph is undirected and A = A^T
        ag = broad_func(self, graph, ampbyp, grad_output)

        grad_input = ag.mm(weight.t())
        grad_weight = outer_product(inputs.t(), ag, self.group)

        return None, None, None, grad_input, grad_weight
