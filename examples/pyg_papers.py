"""Multi-node multi-GPU example on ogbn-papers100m.

Example way to run using srun:
srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> \
--container-name=cont --container-image=<image_url> \
--container-mounts=/ogb-papers100m/:/workspace/dataset
python3 path_to_script.py
"""
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

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCN, GraphSAGE


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

    kwargs = dict(
        data=data,
        batch_size=1024,
        num_workers=get_num_workers(),
        num_neighbors=[15, 10, 5],
    )

    print(f"before neighbor loader", flush=True)
    train_loader = NeighborLoader(
        input_nodes=split_idx['train'],
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    print(f"after neighbor loader", flush=True)
    # val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
    test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)
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
    for epoch in range(1, 7):
        model.train()
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()
            batch = batch.to(device)
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, y)
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
            for j, batch in enumerate(loader):
                if num_steps is not None and j >= num_steps:
                    break
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                y = batch.y[:batch.batch_size].view(-1).to(torch.long)
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

    os.environ["MASTER_ADDR"] = "nid008460"
    os.environ["MASTER_PORT"] = "1234"
    torch.distributed.init_process_group("nccl")
    rank = dist.get_rank()
    nprocs = dist.get_world_size()
    assert dist.is_initialized(), "Distributed cluster not initialized"
    dataset = PygNodePropPredDataset(name='ogbn-papers100M', root="/global/u1/a/alokt/data")

    device = torch.device(f'cuda:{rank % 1}')
    # data = Data()
    # data.y = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/label/label.pt")
    # data.y = data.y.squeeze().to(device)
    # train_idx = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/index/train_idx.pt")
    # train_idx = train_idx.to(device)
    # test_idx = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/index/test_idx.pt")
    # test_idx = test_idx.to(device)

    # adj_matrix_crows = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/csr/indptr_selfloop.pt")
    # adj_matrix_cols = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/csr/indices_selfloop.pt")
    # 
    # adj_matrix_vals = torch.FloatTensor(adj_matrix_cols.size(0)).fill_(1)
    # g_loc = torch.sparse_csr_tensor(adj_matrix_crows, adj_matrix_cols, adj_matrix_vals)
    # edge_index = g_loc.to_sparse_coo()._indices()
    # data.edge_index = edge_index
    # print(f"data.edge_index: {data.edge_index}", flush=True)
    # print(f"data.edge_index.min: {data.edge_index.min()}", flush=True)
    # print(f"data.edge_index.max: {data.edge_index.max()}", flush=True)
    # print(f"train_idx: {train_idx}", flush=True)
    # print(f"train_idx.min: {train_idx.min()}", flush=True)
    # print(f"train_idx.max: {train_idx.max()}", flush=True)

    # inputs = torch.load("/global/u1/a/alokt/data/ogbn_papers100M/feat/feature.pt")
    # data.x = inputs
    # print(f"inputs: {inputs} inputs.size: {inputs.size()} inputs.dtype {inputs.dtype}", flush=True)
    # num_features = inputs.size(1)
    # num_classes = 172
    # data.num_features = num_features

    split_idx = dataset.get_idx_split()
    print(f"split_idx['train'].size: {split_idx['train'].size()}", flush=True)
    print(f"split_idx['test'].size: {split_idx['test'].size()}", flush=True)
    # split_idx = dict()
    # split_idx["train"] = train_idx
    # split_idx["test"] = test_idx
    # model = GCN(dataset.num_features, 256, 2, dataset.num_classes)
    model = GraphSAGE(dataset.num_features, 256, 3, dataset.num_classes)
    acc = Accuracy(task="multiclass", num_classes=dataset.num_classes)
    # model = GCN(num_features, 256, 2, num_classes)
    # acc = Accuracy(task="multiclass", num_classes=num_classes)
    data = dataset[0]
    data.y = data.y.reshape(-1)
    run(nprocs, data, split_idx, model, acc, wall_clock_start)
