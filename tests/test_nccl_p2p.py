import argparse
import os
import torch
import torch.distributed as dist
import socket

def start_time(timer):
    timer.record()

def stop_time(start_timer, stop_timer):
    stop_timer.record()
    torch.cuda.synchronize()
    return start_timer.elapsed_time(stop_timer)

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

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    # # send/recv test
    # if rank == 0:
    #     x = torch.cuda.IntTensor(5).fill_(3) # vector of size 5, set to all 3's
    #     dist.send(x, dst=1)
    # elif rank == 1:
    #     y = torch.cuda.IntTensor(5).fill_(2) # vector of size 5, set to all 2's
    #     print(f"y_before: {y}")
    #     dist.recv(y, src=0)
    #     print(f"y_after: {y}")

    # isend/irecv test
    # buffer_size = 344546571 # spgemm
    # buffer_size = 37135566  # sa-spgemm
    buffer_size = 37135566 // 4  # sa-spgemm single isend
    # Communicate once w/o timing
    if rank == 0:
        recv_objs = []
        recv_tens = torch.cuda.LongTensor(buffer_size).fill_(2)

        print(f"before recv_tens: {recv_tens}")
        # dist.irecv(recv_tens, src=1).wait()
        dist.recv(recv_tens, src=1)
        print(f"after recv_tens: {recv_tens}")
    else:
        y = torch.cuda.LongTensor(buffer_size).fill_(3) # vector of size 5, set to all 2's
        # dist.isend(y, dst=0).wait()
        dist.send(y, dst=0)

    if rank == 0:
        recv_objs = []
        recv_tens = torch.cuda.LongTensor(buffer_size).fill_(5)

        print(f"before recv_tens: {recv_tens}")
        # dist.irecv(recv_tens, src=1).wait()
        dist.recv(recv_tens, src=1)
        print(f"after recv_tens: {recv_tens}")
    else:
        y = torch.cuda.LongTensor(buffer_size).fill_(6) # vector of size 5, set to all 2's
        start_time(start_timer)
        # dist.isend(y, dst=0).wait()
        dist.send(y, dst=0)
        seconds = stop_time(start_timer, stop_timer) / 1000
        gb_count = (buffer_size * 8) / 2**30
        bw = gb_count / seconds
        print(f"gb: {gb_count}GB time: {seconds}s bw: {bw}GB/s")
        print(f"time(ms): {seconds * 1000}")
    
    # # gather test
    # if rank > 0:
    #     x = torch.cuda.IntTensor(5).fill_(rank + 1)
    #     print(f"x: {x}")
    #     dist.gather(x, dst=0)
    # else:
    #     x = torch.cuda.IntTensor(5).fill_(rank + 1)
    #     gather_list = []
    #     for i in range(size):
    #         gather_list.append(torch.cuda.IntTensor(5).fill_(0))

    #     print(f"before gather_list: {gather_list}")
    #     dist.gather(x, gather_list)
    #     print(f"after gather_list: {gather_list}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCCL P2P test')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')

    args = parser.parse_args()
    test_nccl(args)
