import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    group = dist.new_group([0, 1])
    tensors = [None, None]
    tensors[0] = torch.zeros(2, 2)
    tensors[1] = torch.ones(2, 2)

    output_tensor = torch.ones(2, 2)
    if rank == 0:
        output_tensor += 1
        dist.scatter(output_tensor, src=0, scatter_list=tensors, group=group)
    else:
        dist.scatter(output_tensor, src=0, group=group)

    if rank == 1:
        output_tensor += 5
        print(output_tensor)

    if rank == 0:
        print(tensors[1])

    print('Rank ', rank, ' has data ', output_tensor)

def run_p2p(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
