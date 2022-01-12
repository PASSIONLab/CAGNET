from cagnet.partitionings import Partitioning
import torch
import torch.nn.functional as F
import torch.distributed as dist

def relu(x, partitioning):
    if partitioning == Partitioning.ONED or partitioning == Partitioning.ONE5D:
        return F.relu(x)

def log_softmax(x, partitioning, dim=1):
    if partitioning == Partitioning.ONED or partitioning == Partitioning.ONE5D:
        return F.log_softmax(x, dim)

def cross_entropy(logits_rank, labels_rank, node_count, partitioning, rank, group, size):
    if partitioning == Partitioning.ONED or partitioning == Partitioning.ONE5D:
        loss = F.cross_entropy(logits_rank, labels_rank, reduction="sum") 

        loss_recv = []
        for i in range(size):
            loss_recv.append(torch.cuda.FloatTensor(loss.size()))
        dist.all_gather(loss_recv, loss, group)
        loss_recv[rank] = loss
        loss = sum(loss_recv) / node_count
        return loss
