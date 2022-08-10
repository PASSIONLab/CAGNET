import argparse
import os
import torch
import torch.distributed as dist
import socket

def test_nccl(args):
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}")
    print(f"backend: {dist.get_backend()}")

    # # send/recv test
    # if rank == 0:
    #     x = torch.cuda.IntTensor(5).fill_(3) # vector of size 5, set to all 3's
    #     dist.send(x, dst=1)
    # elif rank == 1:
    #     y = torch.cuda.IntTensor(5).fill_(2) # vector of size 5, set to all 2's
    #     print(f"y_before: {y}")
    #     dist.recv(y, src=0)
    #     print(f"y_after: {y}")

    # # isend/irecv test
    # if rank == 0:
    #     x = torch.cuda.IntTensor(rank + 1).fill_(0) # vector of size 5, set to all 3's
    #     recv_objs = []
    #     recv_tens = []
    #     for i in range(size):
    #         recv_tens.append(torch.cuda.IntTensor(i + 1).fill_(0))

    #     print(f"before recv_tens: {recv_tens}")
    #     for i in range(1, size):
    #         print(f"i: {i}")
    #         recv_objs.append(dist.irecv(recv_tens[i], src=i))
    #     # map(lambda x:x.wait(), recv_objs)
    #     for obj in recv_objs:
    #         obj.wait()
    #     print(f"after recv_tens: {recv_tens}")
    # else:
    #     y = torch.cuda.IntTensor(rank + 1).fill_(rank) # vector of size 5, set to all 2's
    #     print(f"rank: {rank} y: {y}")
    #     send_obj = dist.isend(y, dst=0)
    #     send_obj.wait()
    
    # gather test
    if rank > 0:
        x = torch.cuda.IntTensor(5).fill_(rank + 1)
        print(f"x: {x}")
        dist.gather(x, dst=0)
    else:
        x = torch.cuda.IntTensor(5).fill_(rank + 1)
        gather_list = []
        for i in range(size):
            gather_list.append(torch.cuda.IntTensor(5).fill_(0))

        print(f"before gather_list: {gather_list}")
        dist.gather(x, gather_list)
        print(f"after gather_list: {gather_list}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCCL P2P test')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')

    args = parser.parse_args()
    test_nccl(args)
