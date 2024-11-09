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

    os.environ["MASTER_ADDR"] = args.hostname # hostname of rank 0's node
    os.environ["MASTER_PORT"] = "1234"

    print(f"hostname: {socket.gethostname()}", flush=True)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    torch.cuda.set_device(rank % args.gpu)
    print(f"os.cpu_count: {os.cpu_count()}", flush=True)
    print(f"cpus_per_task: {os.environ['SLURM_CPUS_PER_TASK']}", flush=True)

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

    # # isend/irecv test
    # # buffer_size = 344546571 # spgemm
    # # buffer_size = 37135566  # sa-spgemm
    # buffer_size = 37135566 // 4  # sa-spgemm single isend
    # # communicate once w/o timing
    # if rank == 0:
    #     recv_objs = []
    #     recv_tens = torch.cuda.LongTensor(buffer_size).fill_(2)

    #     print(f"before recv_tens: {recv_tens}")
    #     # dist.irecv(recv_tens, src=1).wait()
    #     dist.recv(recv_tens, src=1)
    #     print(f"after recv_tens: {recv_tens}")
    # else:
    #     y = torch.cuda.LongTensor(buffer_size).fill_(3) # vector of size 5, set to all 2's
    #     # dist.isend(y, dst=0).wait()
    #     dist.send(y, dst=0)

    # if rank == 0:
    #     recv_objs = []
    #     recv_tens = torch.cuda.LongTensor(buffer_size).fill_(5)

    #     print(f"before recv_tens: {recv_tens}")
    #     # dist.irecv(recv_tens, src=1).wait()
    #     dist.recv(recv_tens, src=1)
    #     print(f"after recv_tens: {recv_tens}")
    # else:
    #     y = torch.cuda.LongTensor(buffer_size).fill_(6) # vector of size 5, set to all 2's
    #     start_time(start_timer)
    #     # dist.isend(y, dst=0).wait()
    #     dist.send(y, dst=0)
    #     seconds = stop_time(start_timer, stop_timer) / 1000
    #     gb_count = (buffer_size * 8) / 2**30
    #     bw = gb_count / seconds
    #     print(f"gb: {gb_count}GB time: {seconds}s bw: {bw}GB/s")
    #     print(f"time(ms): {seconds * 1000}")
    
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

    # # (sparse) allreduce test
    # # buffer_size = 5
    # buffer_size = 16213041
    # # x_ind = torch.arange(buffer_size).cuda().long()
    # # x_ind = torch.stack((x_ind, x_ind))
    # # x_val = torch.cuda.FloatTensor(buffer_size).fill_(1.0)
    # # x = torch.sparse_coo_tensor(x_ind, x_val, size=(buffer_size, buffer_size))
    # x = torch.cuda.FloatTensor(buffer_size).fill_(3)
    # recv_list = []
    # for i in range(size):
    #     recv_list.append(torch.cuda.FloatTensor(buffer_size))
    # recv_list2 = []
    # for i in range(size):
    #     recv_list2.append(torch.cuda.FloatTensor(buffer_size))
    # # dist.all_reduce(x)
    # dist.all_gather(recv_list, x)
    # dist.all_gather(recv_list2, x)
    # # if rank == 0:
    # #     dist.send(x, 2)
    # #     dist.recv(recv_list[0], 2)
    # #     x = x + recv_list[0]
    # # elif rank == 2:
    # #     dist.recv(recv_list[0], 0)
    # #     dist.send(x, 0)
    # #     x = x + recv_list[0]

    # # if rank == 1:
    # #     dist.send(x, 3)
    # #     dist.recv(recv_list[0], 3)
    # #     x = x + recv_list[0]
    # # elif rank == 3:
    # #     dist.recv(recv_list[0], 1)
    # #     dist.send(x, 1)
    # #     x = x + recv_list[0]

    # # if rank == 0:
    # #     dist.send(x, 1)
    # #     dist.recv(recv_list[0], 1)
    # #     x = x + recv_list[0]
    # # elif rank == 1:
    # #     dist.recv(recv_list[0], 0)
    # #     dist.send(x, 0)
    # #     x = x + recv_list[0]

    # # if rank == 2:
    # #     dist.send(x, 3)
    # #     dist.recv(recv_list[0], 3)
    # #     x = x + recv_list[0]
    # # elif rank == 3:
    # #     dist.recv(recv_list[0], 2)
    # #     dist.send(x, 2)
    # #     x = x + recv_list[0]

    # start_time(start_timer)
    # # dist.all_reduce(x)
    # dist.all_gather(recv_list, x)
    # dist.all_gather(recv_list2, x)
    # # if rank == 0:
    # #     dist.send(x, 1)
    # #     dist.recv(recv_list[0], 1)
    # # elif rank == 1:
    # #     dist.recv(recv_list[0], 0)
    # #     dist.send(x, 0)
    # # if rank == 0:
    # #     dist.send(x, 2)
    # #     dist.recv(recv_list[0], 2)
    # #     x = x + recv_list[0]
    # # elif rank == 2:
    # #     dist.recv(recv_list[0], 0)
    # #     dist.send(x, 0)
    # #     x = x + recv_list[0]

    # # if rank == 1:
    # #     dist.send(x, 3)
    # #     dist.recv(recv_list[0], 3)
    # #     x = x + recv_list[0]
    # # elif rank == 3:
    # #     dist.recv(recv_list[0], 1)
    # #     dist.send(x, 1)
    # #     x = x + recv_list[0]

    # # if rank == 0:
    # #     dist.send(x, 1)
    # #     dist.recv(recv_list[0], 1)
    # #     x = x + recv_list[0]
    # # elif rank == 1:
    # #     dist.recv(recv_list[0], 0)
    # #     dist.send(x, 0)
    # #     x = x + recv_list[0]

    # # if rank == 2:
    # #     dist.send(x, 3)
    # #     dist.recv(recv_list[0], 3)
    # #     x = x + recv_list[0]
    # # elif rank == 3:
    # #     dist.recv(recv_list[0], 2)
    # #     dist.send(x, 2)
    # #     x = x + recv_list[0]
    # seconds = stop_time(start_timer, stop_timer) / 1000
    # gb_count = (buffer_size * 8) / 2**30
    # bw = gb_count / seconds
    # print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    # print(f"time(ms): {seconds * 1000}")

    # # scatter test
    # buffer_size = 5
    # x = torch.cuda.FloatTensor(buffer_size + 3).fill_(2)
    # y = torch.cuda.FloatTensor(buffer_size).fill_(3)
    # if rank == 0:
    #     scatter_list = [x, y]
    # else:
    #     scatter_list = None

    # if rank == 0:
    #     z = torch.cuda.FloatTensor(buffer_size + 3)
    # else:
    #     z = torch.cuda.FloatTensor(buffer_size)
    # dist.scatterv(z, scatter_list, src=0)

    # print(f"x: {x}")
    # print(f"y: {y}")
    # print(f"z: {z}")

    # # batch isend vs blocking send test
    # dist.barrier()
    # buffer_size = 1 * 2**30
    # run_count = 2
    # torch.cuda.profiler.cudart().cudaProfilerStart()
    # print(f"blocking send")
    # torch.cuda.nvtx.range_push(f"nvtx-blocksend-rank{rank}")
    # for r in range(run_count):
    #     tensors = []
    #     for i in range(size - 1):
    #         tensors.append(torch.cuda.FloatTensor(buffer_size).fill_(i + 1))

    #     if rank == 0:
    #         if r == run_count - 1:
    #             start_time(start_timer)

    #         for i in range(size - 1):
    #             dist.send(tensors[i], dst=i+1)
    #         torch.cuda.synchronize()
    #         recv_tensor = torch.cuda.FloatTensor(buffer_size).fill_(0)

    #         if r == run_count - 1:
    #             seconds = stop_time(start_timer, stop_timer) / 1000
    #             gb_count = (buffer_size * 4) / 2**30
    #             bw = gb_count / seconds
    #             print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    #             print(f"time(ms): {seconds * 1000}")
    #     else:
    #         recv_tensor = torch.cuda.FloatTensor(buffer_size)
    #         dist.recv(recv_tensor, src=0)
    #     dist.barrier()
    # dist.barrier()
    # torch.cuda.nvtx.range_pop()

    # dist.barrier()
    # print(f"batch send", flush=True)
    # buffer_size = 2000 * 2**20 // size
    # run_count = 2
    # torch.cuda.nvtx.range_push(f"nvtx-batchsend-rank{rank}")
    # for r in range(run_count):
    #     tensors = []
    #     for i in range(size - 1):
    #         tensors.append(torch.cuda.FloatTensor(buffer_size).fill_(i + 1))

    #     send_ops = []
    #     for i in range(size - 1):
    #         send_ops.append(dist.P2POp(dist.isend, tensors[i], i + 1))

    #     if rank == 0:
    #         if r == run_count - 1:
    #             start_time(start_timer)

    #         reqs = dist.batch_isend_irecv(send_ops)
    #         for req in reqs:
    #             req.wait()
    #         recv_tensor = torch.cuda.FloatTensor(buffer_size).fill_(0)
    #         torch.cuda.synchronize()

    #         if r == run_count - 1:
    #             seconds = stop_time(start_timer, stop_timer) / 1000
    #             gb_count = (buffer_size * 4) / 2**30
    #             bw = gb_count / seconds
    #             print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    #             print(f"time(ms): {seconds * 1000}")
    #     else:
    #         recv_tensor = torch.cuda.FloatTensor(buffer_size)
    #         recv_ops = []
    #         recv_ops.append(dist.P2POp(dist.irecv, tensors[i], 0))
    #         reqs = dist.batch_isend_irecv(recv_ops)
    #         for req in reqs:
    #             req.wait()
    #         # dist.recv(recv_tensor, src=0)
    #     dist.barrier()
    # dist.barrier()
    # torch.cuda.nvtx.range_pop()

    # print(f"scatter")
    # torch.cuda.nvtx.range_push(f"nvtx-scatter-rank{rank}")
    # for r in range(run_count):
    #     if rank == 0:
    #         tensors = []
    #         for i in range(size):
    #             tensors.append(torch.cuda.FloatTensor(buffer_size).fill_(i))
    #     else:
    #         tensors = None

    #     recv_tensor = torch.cuda.FloatTensor(buffer_size)

    #     if rank == 0 and r == run_count - 1:
    #         start_time(start_timer)

    #     dist.scatter(recv_tensor, tensors, src=0)
    #     torch.cuda.synchronize()

    #     if rank == 0 and r == run_count - 1:
    #         seconds = stop_time(start_timer, stop_timer) / 1000
    #         gb_count = (buffer_size * 4) / 2**30
    #         bw = gb_count / seconds
    #         print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    #         print(f"time(ms): {seconds * 1000}")
    #     dist.barrier()
    # dist.barrier()
    # torch.cuda.nvtx.range_pop()

    # print(f"nonblocking isends")
    # torch.cuda.nvtx.range_push(f"nvtx-nonblockisend-rank{rank}")
    # for r in range(run_count):
    #     tensors = []
    #     streams = []
    #     for i in range(size - 1):
    #         tensors.append(torch.cuda.FloatTensor(buffer_size).fill_(i + 1))
    #         streams.append(torch.cuda.Stream())

    #     if rank == 0:
    #         if r == run_count - 1:
    #             start_time(start_timer)

    #         reqs = []
    #         for i in range(size - 1):
    #             with torch.cuda.stream(streams[i]):
    #                 reqs.append(dist.isend(tensors[i], dst=i+1))
    #         for req in reqs:
    #             with torch.cuda.stream(streams[i]):
    #                 req.wait()
    #         torch.cuda.synchronize()
    #         recv_tensor = torch.cuda.FloatTensor(buffer_size).fill_(0)

    #         if r == run_count - 1:
    #             seconds = stop_time(start_timer, stop_timer) / 1000
    #             gb_count = (buffer_size * 4) / 2**30
    #             bw = gb_count / seconds
    #             print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    #             print(f"time(ms): {seconds * 1000}")
    #     else:
    #         recv_tensor = torch.cuda.FloatTensor(buffer_size)
    #         dist.recv(recv_tensor, src=0)
    
    # torch.cuda.nvtx.range_pop()
    # torch.cuda.profiler.cudart().cudaProfilerStop()

    # print(f"alltoall", flush=True)
    # buffer_size = 1 * 2**30 // size
    # run_count = 2
    # for r in range(run_count):
    #     tensors = []
    #     for i in range(size):
    #         tensors.append(torch.cuda.FloatTensor(buffer_size).fill_(i))
    #     tensors[-1] = torch.cuda.FloatTensor(0).fill_(i)

    #     recv_tensors = []
    #     for i in range(size):
    #         if rank == size - 1:
    #             recv_tensors.append(torch.cuda.FloatTensor(0))
    #         else:
    #             recv_tensors.append(torch.cuda.FloatTensor(buffer_size))

    #     if r == run_count - 1 and rank == 0:
    #         start_time(start_timer)

    #     # dist.scatter(recv_tensor, tensors, src=0)
    #     dist.all_to_all(recv_tensors, tensors)
    #     torch.cuda.synchronize()

    #     if r == run_count - 1 and rank == 0:
    #         seconds = stop_time(start_timer, stop_timer) / 1000
    #         gb_count = (buffer_size * 4 * size) / 2**30
    #         bw = gb_count / seconds
    #         print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    #         print(f"time(ms): {seconds * 1000}")
    #     dist.barrier()

    # dist.barrier()
    # print(f"batched isend/irecv alltoall", flush=True)
    # buffer_size = 1 * 2**30 // size
    # # buffer_size = ((66 // 4) * 2**20 // 2) // (size - 1)
    # run_count = 3
    # for r in range(run_count):
    #     send_tensors = [None] * size
    #     recv_tensors = [None] * size
    #     elem_count = 0
    #     for i in range(size):
    #         if i != rank:
    #             elem_count += 2 * buffer_size
    #             send_tensors[i] = torch.cuda.FloatTensor(buffer_size).fill_(i + 1)
    #             recv_tensors[i] = torch.cuda.FloatTensor(buffer_size).fill_(0)

    #     ops = []
    #     for i in range(size):
    #         if i != rank:
    #             ops.append(dist.P2POp(dist.isend, send_tensors[i], i))
    #             ops.append(dist.P2POp(dist.irecv, recv_tensors[i], i))

    #     if r == run_count - 1:
    #         start_time(start_timer)
    #     reqs = dist.batch_isend_irecv(ops)
    #     for req in reqs:
    #         req.wait()
    #     torch.cuda.synchronize()
    #     dist.barrier()

    #     if r == run_count - 1 and rank == 0:
    #         seconds = stop_time(start_timer, stop_timer) / 1000
    #         gb_count = (buffer_size * size * 4) / 2**30 # assumes buffer_size is per process
    #         bw = gb_count / seconds
    #         print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
    #         print(f"time(ms): {seconds * 1000}")
    #         print(f"elem_count: {elem_count}")

    print(f"p2p cross-node send", flush=True)
    buffer_size = 10 * 2**30
    # buffer_size = ((66 // 4) * 2**20 // 2) // (size - 1)
    if rank == 0:
        send_tensor = torch.cuda.FloatTensor(buffer_size).fill_(1.0)
    elif rank == size - 1:
        recv_tensor = torch.cuda.FloatTensor(buffer_size).fill_(1.0)

    dist.barrier()
    if rank == size - 1:
        start_time(start_timer)
    
    if rank == 0:
        dist.send(send_tensor, dst=size-1)
    elif rank == size - 1:
        dist.recv(recv_tensor, src=0)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == size - 1:
        seconds = stop_time(start_timer, stop_timer) / 1000
        gb_count = (buffer_size * 4) / 2**30 # assumes buffer_size is per process
        bw = gb_count / seconds
        print(f"gb: {gb_count} GB time: {seconds}s bw: {bw}GB/s")
        print(f"time(ms): {seconds * 1000}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCCL P2P test')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument("--gpu", type=int, default=4,
                        help="gpus per node")

    args = parser.parse_args()
    test_nccl(args)
