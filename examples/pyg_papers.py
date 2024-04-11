"""Multi-node multi-GPU example on ogbn-papers100m.

Example way to run using srun:
srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> \
--container-name=cont --container-image=<image_url> \
--container-mounts=/ogb-papers100m/:/workspace/dataset
python3 path_to_script.py
"""
import gc
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, Dataset
from torchmetrics import Accuracy
from torch_geometric.typing import SparseTensor

from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.nn import GCN, GraphSAGE, SAGEConv

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        # m.bias.data.fill_(0.01)

class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embed_dim,
                 num_layers: int,
                 act: str = 'ReLU',
                 bn: bool = False,
                 end_up_with_fc=False,
                 bias=True):
        super(MLP, self).__init__()
        self.module_list = []

        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = embed_dim if i == num_layers - 1 else hidden_dim
            self.module_list.append(torch.nn.Linear(d_in, d_out, bias=bias))
            if end_up_with_fc:
                continue
            if bn:
                self.module_list.append(torch.nn.BatchNorm1d(d_out))
            self.module_list.append(getattr(torch.nn, act)(True))
        self.module_list = torch.nn.Sequential(*self.module_list)

    def reset_parameters(self):
        for x in self.module_list:
            if hasattr(x, "reset_parameters"):
                x.reset_parameters()

    def forward(self, x):
        return self.module_list(x)

class SAGEResInception(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        conv_layer = SAGEConv
        kwargs = dict(bias=False)
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.res_linears = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.res_linears.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.res_linears.append(torch.nn.Identity())
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, **kwargs))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.res_linears.append(torch.nn.Identity())

        self.mlp = MLP(in_channels + hidden_channels * (num_layers),
                       2*out_channels, out_channels,
                       num_layers=2, bn=True, end_up_with_fc=True,
                       act='LeakyReLU')
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)
        for x in self.res_linears:
            if isinstance(x, torch.nn.Linear):
                x.reset_parameters()
        for x in self.bns:
            x.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, _x, adjs):
        _x = _x.to(torch.float)
        collect = []
        end_size = adjs[-1][-1][1]
        x = F.dropout(_x, p=0.1, training=self.training)
        collect.append(x[:end_size])
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((F.dropout(x, p=0.1, training=self.training),
                               F.dropout(x_target, p=0.1,
                                         training=self.training)), edge_index)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            collect.append(x[:end_size])
            x += self.res_linears[i](x_target)
        return torch.log_softmax(self.mlp(torch.cat(collect, -1)), dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                # x = self.convs[i]((x, x_target), edge_index)
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # xs.append(x)
                xs.append(x[:batch_size].cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        if self.num_layers > 1:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(self.num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, out_channels))

    def forward(self, x, adjs):
        if self.num_layers == 1:
            adjs = [adjs]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x.float(), x_target.float()), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x.float(), x_target.float()), edge_index)
                # x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # xs.append(x)
                xs.append(x[:batch_size].cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

def get_num_workers() -> int:
    num_workers = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_workers = len(os.sched_getaffinity(0)) // 2
        except Exception:
            pass
    if num_workers is None:
        num_workers = os.cpu_count() // 2
    return num_workers


def run(world_size, data, split_idx, model, acc, wall_clock_start):
    local_id = int(os.environ['LOCAL_RANK']) % 1
    rank = torch.distributed.get_rank()
    print(f"local_rank: {local_id}", flush=True)
    torch.cuda.set_device(local_id)
    device = torch.device(local_id)
    print(f"after setting local_id: {local_id}", flush=True)
    if rank == 0:
        print(f'Using {nprocs} GPUs...', flush=True)

    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    # split_idx['valid'] = split_idx['valid'].split(
    #     split_idx['valid'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['test'] = split_idx['test'].split(
        split_idx['test'].size(0) // world_size, dim=0)[rank].clone()

    print(f"before ddp device: {device} local_id: {local_id}", flush=True)
    model = DistributedDataParallel(model.to(device), device_ids=[local_id])
    print(f"after ddp", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=5e-4)

    # kwargs = dict(
    #     data=data,
    #     batch_size=1024,
    #     num_workers=get_num_workers(),
    #     num_neighbors=[15, 10],
    # )

    print(f"before neighbor loader", flush=True)
    # train_loader = NeighborLoader(
    #     input_nodes=split_idx['train'],
    #     shuffle=True,
    #     drop_last=True,
    #     **kwargs,
    # )
    num_nodes = data.x.size(0)
    value = torch.arange(data.edge_index.size(1))
    graph_sparsetens = SparseTensor(row=data.edge_index[0,:], col=data.edge_index[1,:],
                              value=value,
                              is_sorted=True,
                              sparse_sizes=(num_nodes, num_nodes))
    train_loader = NeighborSampler(graph_sparsetens, node_idx=split_idx["train"],
				       sizes=[15, 10], batch_size=1024,
				       shuffle=False, drop_last=True, num_workers=get_num_workers())
    print(f"after neighbor loader", flush=True)
    # val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
    # test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)
    test_loader = NeighborSampler(graph_sparsetens, node_idx=split_idx["test"],
				       sizes=[20, 20], batch_size=1024,
				       shuffle=False, drop_last=True, num_workers=get_num_workers())
    print(f"after loaders", flush=True)

    val_steps = 1000
    warmup_steps = 100
    acc = acc.to(device)
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time)=", prep_time,
              "seconds")
        print("Beginning training...")

    print(f"len(train_loader): {len(train_loader)}", flush=True)
    for epoch in range(1, 6):
        model.train()
        # for i, batch in enumerate(train_loader):
        for i, (batch_size, n_id, adjs) in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()
            # batch = batch.to(device)
            if model.module.num_layers > 1:
                adjs = [adj.to(device) for adj in adjs]
            else:
                adjs = adjs.to(device)
            optimizer.zero_grad()
            # y = batch.y[:batch.batch_size].view(-1).to(torch.long)
            # out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = data.y[n_id[:batch_size]].to(device).long()
            out = model(data.x[n_id].to(device), adjs)
            loss = F.cross_entropy(out[:batch_size,:], y)
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                print(f'Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}')

        dist.barrier()
        torch.cuda.synchronize()
        if rank == 0:
            sec_per_iter = (time.time() - start) / (i + 1 - warmup_steps)
            print(f"Avg Training Iteration Time: {sec_per_iter:.6f} s/iter")

        @torch.no_grad()
        def test(loader: NeighborLoader, num_steps: Optional[int] = None):
            model.eval()
            # for j, batch in enumerate(loader):
            for j, (batch_size, n_id, adjs) in enumerate(train_loader):
                if num_steps is not None and j >= num_steps:
                    break
                # batch = batch.to(device)
                # out = model(batch.x, batch.edge_index)[:batch.batch_size]
                # y = batch.y[:batch.batch_size].view(-1).to(torch.long)
                if model.module.num_layers > 1:
                    adjs = [adj.to(device) for adj in adjs]
                else:
                    adjs = adjs.to(device)
                y = data.y[n_id[:batch_size]].to(device).long()
                out = model(data.x[n_id].to(device), adjs)
                acc(out, y)
            acc_sum = acc.compute()
            return acc_sum

        # eval_acc = test(val_loader, num_steps=val_steps)
        # if rank == 0:
        #     print(f"Val Accuracy: {eval_acc:.4f}%", )

        acc.reset()
        dist.barrier()

    test_acc = test(test_loader)
    if rank == 0:
        print(f"Test Accuracy: {test_acc:.4f}%", )

    dist.barrier()
    acc.reset()
    torch.cuda.synchronize()

    if rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")


if __name__ == '__main__':
    wall_clock_start = time.perf_counter()
    # Setup multi-node:
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    os.environ["MASTER_ADDR"] = "nid008196"
    os.environ["MASTER_PORT"] = "1234"
    torch.distributed.init_process_group("nccl")
    rank = dist.get_rank()
    nprocs = dist.get_world_size()
    assert dist.is_initialized(), "Distributed cluster not initialized"
    # dataset = PygNodePropPredDataset(name='ogbn-papers100M', root="/global/u1/a/alokt/data")

    device = torch.device(f'cuda:{rank % 1}')
    data = Data()
    graph_crows = torch.load("/global/homes/a/alokt/SALIENT_plusplus_artifact/dataset/ogbn-papers100M/rowptr.pt")
    graph_cols = torch.load("/global/homes/a/alokt/SALIENT_plusplus_artifact/dataset/ogbn-papers100M/col.pt")
    graph_vals = torch.FloatTensor(graph_cols.size(0)).fill_(1.0)
    num_nodes = graph_crows.size(0) - 1
    graph = torch.sparse_csr_tensor(graph_crows, graph_cols, graph_vals)
    data.edge_index = graph.to_sparse_coo()._indices()
    data.x = torch.load("/global/homes/a/alokt/SALIENT_plusplus_artifact/dataset/ogbn-papers100M/x.pt")
    # data.x = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/feat/feature.pt")
    data.y = torch.load("/global/homes/a/alokt/SALIENT_plusplus_artifact/dataset/ogbn-papers100M/y.pt")
    split_idx = torch.load("/global/homes/a/alokt/SALIENT_plusplus_artifact/dataset/ogbn-papers100M/split_idx.pt")

    # split_idx = dataset.get_idx_split()
    print(f"split_idx['train'].size: {split_idx['train'].size()}", flush=True)
    print(f"split_idx['test'].size: {split_idx['test'].size()}", flush=True)
    # split_idx = dict()
    # split_idx["train"] = train_idx
    # split_idx["test"] = test_idx
    # model = GCN(dataset.num_features, 256, 2, dataset.num_classes)
    # model = GraphSAGE(dataset.num_features, 256, 2, dataset.num_classes)
    model = SAGE(128, 256, 172, num_layers=2).to(rank)
    # model = SAGEResInception(128, 256, 172, num_layers=3).to(rank)
    acc = Accuracy(task="multiclass", num_classes=172)
    # model = GCN(num_features, 256, 2, num_classes)
    # acc = Accuracy(task="multiclass", num_classes=num_classes)
    # data = dataset[0]

    # print(f"before casting", flush=True)
    # if rank == 0:
    #     x = input()
    # dist.barrier()
    # edge_index = data.edge_index
    # src = edge_index[0,:]
    # dst = edge_index[1,:]
    # reverse_edges = torch.stack((dst, src))
    # data.edge_index = torch.cat((edge_index, reverse_edges), dim=1)
    # x = data.x.to(torch.float16)
    # del data.x
    # gc.collect()
    # data.x = x
    # print(f"after casting", flush=True)
    # if rank == 0:
    #     x = input()
    # dist.barrier()
    # del reverse_edges
    # del edge_index
    # gc.collect()
    # print(f"after collect", flush=True)
    # if rank == 0:
    #     x = input()
    # dist.barrier()
    print(f"data.edge_index: {data.edge_index}", flush=True)
    print(f"data.x: {data.x}", flush=True)
    print(f"data.edge_index.dtype: {data.edge_index.dtype}", flush=True)
    print(f"data.x.dtype: {data.x.dtype}", flush=True)
    print(f"data.edge_index.size: {data.edge_index.size()}", flush=True)
    data.y = data.y.reshape(-1)
    run(nprocs, data, split_idx, model, acc, wall_clock_start)
