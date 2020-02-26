"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    """ Distributed function to be implemented later. """
    print("hi from rank: " + str(rank))

def init_process(rank, size, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    print(os.environ['WORLD_SIZE'])
    print(os.environ['RANK'])
    print(os.environ['SLURM_NODEID'])
    dist.init_process_group(init_method='env://', backend=backend)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    print(str(rank) + " " + str(n_ranks))
    # for rank in range(size):
    #     p = Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

# import argparse
# import socket
# import torch.distributed as dist
# 
# parser = argparse.ArgumentParser(description='Process some integers.')
# 
# parser.add_argument('--distributed', action='store_true', help='enables distributed processes')
# parser.add_argument('--local_rank', default=0, type=int, help='number of distributed processes')
# parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
# 
# def main():
#     opt = parser.parse_args()
#     if opt.distributed:
#         dist.init_process_group(backend=opt.dist_backend, init_method='env://')
# 
#     print("Initialized Rank:", dist.get_rank())
#     print("hostname: " + str(socket.gethostname()))
# 
# if __name__ == '__main__':
#     main()
