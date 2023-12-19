import math
import torch
import torch.distributed as dist
import torch.nn as nn
import time

from cagnet.partitionings import Partitioning
from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

def stop_time(self, range_name, start, barrier=True):
    barrier=False
    if self.timers and self.epoch > 0:
        torch.cuda.synchronize()
        self.timings[range_name] += time.time() - start
    else:
        return 0.0
    if barrier:
        start = time.time()
        dist.barrier()
        self.timings["barrier"] += time.time() - start
        
def broad_func_oned(self, graph, ampbyp, inputs):
    """ # this is the old function
    
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
        start = time.time()
        dist.broadcast(inputs_recv, src=i, group=self.group)
        stop_time(self, "bcast", start)
        start = time.time()
        spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(), 
                        ampbyp[i].values(), ampbyp[i].size(0), 
                        ampbyp[i].size(1), inputs_recv, z_loc)
        stop_time(self, "spmm_gpu", start)
    return z_loc
    """

    start = time.time()
    z_loc = torch.cuda.FloatTensor(ampbyp[0].size(0), inputs.size(1), device=self.device).fill_(1)
    
    row_indices_send = self.row_indices_send
    row_data_send = [torch.cuda.FloatTensor(device=self.device)]*self.size
    row_indices_recv = self.row_indices_recv
    row_data_recv = [torch.cuda.FloatTensor(device=self.device).resize_(row_indices_send[i].size(0), inputs.size(1)).fill_(0) for i in range(self.size)]
    stop_time(self, "allocate tensors", start, barrier=False)
    
    start = time.time()
    for i in range(self.size):
        row_data_send[i] = inputs[row_indices_recv[i].long(), :]
    stop_time(self, "gather_row_data", start, barrier=False)

    start = time.time()
    dist.all_to_all(row_data_recv, row_data_send, group=self.group)
    stop_time(self, "a2a3", start, barrier=False)

    start = time.time()
    for i in range(self.size):
       inputs_mul = torch.cuda.FloatTensor( device = self.device).resize_(ampbyp[i].size(1), inputs.size(1)).fill_(0)
       inputs_mul[row_indices_send[i]] =  row_data_recv[i]
       spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                        ampbyp[i].values(), ampbyp[i].size(0),
                        ampbyp[i].size(1), inputs_mul, z_loc)
    stop_time(self, "spmm_gpu", start, barrier=False)
    #del inputs_mul
    #torch.cuda.empty_cache()
    return z_loc

def broad_func_one5d(self, graph, ampbyp, inputs):
    """ old code

    
    n_per_proc = math.ceil(float(graph.size(0)) / (self.size / self.replication))

    z_loc = torch.cuda.FloatTensor(ampbyp[0].size(0), inputs.size(1), device=self.device).fill_(0)

    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=self.device).fill_(0)

    rank_c = self.rank // self.replication
    rank_col = self.rank % self.replication

    stages = self.size // (self.replication ** 2)
    if rank_col == self.replication - 1:
        stages = (self.size // self.replication) - (self.replication - 1) * stages
    
    for i in range(stages):
        q = (rank_col * (self.size // (self.replication ** 2)) + i) * self.replication + rank_col

        q_c = q // self.replication

        am_partid = rank_col * (self.size // self.replication ** 2) + i

        if q == self.rank:
            inputs_recv = inputs.clone()
        elif q_c == self.size // self.replication - 1:
            inputs_recv = torch.cuda.FloatTensor(ampbyp[am_partid].size(1), \
                                                    inputs.size(1), \
                                                    device=self.device).fill_(0)

        inputs_recv = inputs_recv.contiguous()
        start = time.time()
        dist.broadcast(inputs_recv, src=q, group=self.col_groups[rank_col])
        stop_time(self, "broadcast", start)
        start = time.time()
        spmm_gpu(ampbyp[am_partid].indices()[0].int(), ampbyp[am_partid].indices()[1].int(), 
                        ampbyp[am_partid].values(), ampbyp[am_partid].size(0), 
                        ampbyp[am_partid].size(1), inputs_recv, z_loc)
        stop_time(self, "spmm_gpu", start)
    z_loc = z_loc.contiguous()
    start = time.time()
    dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=self.row_groups[rank_c])
    stop_time(self, "reduce", start)    
    return z_loc    
    """

    




    # n_per_proc = math.ceil(float(graph.size(0)) / (self.size / self.replication))
    
    """ 1.5d p2p"""
    


    # torch.cuda.synchronize()
    # print(f"before spmm", flush=True)
    # if self.rank == 0:
    #     x = input()
    # dist.barrier()

    start = time.time()
    z_loc = torch.cuda.FloatTensor(ampbyp[0].size(0), inputs.size(1), device=self.device).fill_(0)
    max_ampbyp_size = 0
    for part in ampbyp:
        if part.size(1) > max_ampbyp_size:
            max_ampbyp_size = part.size(1)

    inputs_recv_max = torch.cuda.FloatTensor(max_ampbyp_size, \
                                            inputs.size(1), \
                                            device=self.device).fill_(0)
    rank = self.rank
    rank_c = self.rank // self.replication
    rank_col = self.rank % self.replication
    
    col_procs = list(range(rank_col, self.size, self.replication))
    stages = self.size // (self.replication ** 2)
    if rank_col == self.replication - 1:
        stages = (self.size // self.replication) - (self.replication - 1) * stages
    col_group_length = self.size // self.replication
    row_indices_recv = self.row_indices_recv
    stop_time(self, "allocate/math", start)
    for i in range(stages):
        # if i == 7:
        #     torch.cuda.synchronize()
        #     print(f"before iteration {i}", flush=True)
        #     if self.rank == 0:
        #         x = input()
        #     dist.barrier()
        q = (rank_col * (self.size // (self.replication ** 2)) + i) * self.replication + rank_col
        q_c = q // self.replication
        am_partid = rank_col * (self.size // self.replication ** 2) + i
        unique_cols = self.row_indices_send[q]
        if rank == q:
            send_ops = []
            row_count_sum = 0
            for j in col_procs:
                if rank != j:
                    start = time.time()
                    rows_send = inputs[row_indices_recv[j].long(), :].clone()
                    row_count_sum += rows_send.size(0)
                    stop_time(self, "gather_row_data", start, barrier=False)
                    start = time.time()
                    # dist.send(rows_send, dst=j, group=self.col_groups[rank_col])
                    # send_ops.append(dist.P2POp(dist.isend, rows_send, j, group=self.col_groups[rank_col]))
                    dist.send(rows_send, j, group=self.col_groups[rank_col])
                    stop_time(self, "send_ops", start, barrier=False)
                    del rows_send
    #                print(f"rows_send dim {rows_send.size()}", flush=True)
    #        print(f"pls print {send_ops}", flush=True)
            # print(f"q: {q} row_count_sum: {row_count_sum}", flush=True)
            # if len(send_ops) > 0:
            #     start = time.time()
            #     #print("sendinggg!", flush=True)
            #     reqs = dist.batch_isend_irecv(send_ops)
            #     #print("batch sent", flush=True)
            #     #stop_time(self, "communication", start)
            #     for req in reqs:
            #         req.wait()
            #     stop_time(self, "communication", start, barrier=False)
            #     for op in send_ops:
            #         del op.tensor
            #         del op
            #     send_ops = None
            #     del reqs
            #     torch.cuda.empty_cache()

        # all other ranks receive the inputs
        inputs_recv = []
        if rank!= q:
            rows_recv = torch.cuda.FloatTensor(device=self.device).resize_((unique_cols.size(0), inputs.size(1))).fill_(0)
            start = time.time()
#            print(f"rows_recv dim {rows_recv.size()}", flush=True)
            dist.recv(rows_recv, src=q, group=self.col_groups[rank_col])
            stop_time(self, "communication", start, barrier=False)
            start = time.time()
            # inputs_recv = torch.cuda.FloatTensor(ampbyp[am_partid].size(1), \
            #                                         inputs.size(1), \
            #                                         device=self.device).fill_(0)
            inputs_recv_max.fill_(0)
            inputs_recv = inputs_recv_max[:ampbyp[am_partid].size(1)]
            inputs_recv[unique_cols] = rows_recv
            # del rows_recv
        else:
            start = time.time()
            # inputs_recv = inputs.clone()
            inputs_recv = inputs
        # torch.cuda.synchronize()
        # if i == 7:
        #     print(f"after iteration {i}", flush=True)
        #     if self.rank == 0:
        #         x = input()
        #     dist.barrier()

        # inputs_recv = inputs_recv.contiguous()
    
        spmm_gpu(ampbyp[am_partid].indices()[0].int(), ampbyp[am_partid].indices()[1].int(), 
                        ampbyp[am_partid].values(), ampbyp[am_partid].size(0), 
                        ampbyp[am_partid].size(1), inputs_recv, z_loc)
        stop_time(self, "spmm_gpu", start, barrier=False)

        # del inputs_recv
        
    # torch.cuda.synchronize()
    # print(f"done iterating", flush=True)
    # if self.rank == 0:
    #     x = input()
    # dist.barrier()
    # z_loc = z_loc.contiguous()
    start = time.time()
    dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=self.row_groups[rank_c])
    stop_time(self, "reduce", start, barrier=False)

    # del inputs_recv_max

    return z_loc
    

    """ a2a impl
    """
    

    """
    z_loc = torch.cuda.FloatTensor(ampbyp[0].size(0), inputs.size(1), device=self.device).fill_(0)
    rank = self.rank
    rank_c = self.rank // self.replication
    rank_col = self.rank % self.replication

    rank_c = self.rank // self.replication
    rank_col = self.rank % self.replication

    col_procs = list(range(rank_col, self.size, self.replication))

    stages = self.size // (self.replication ** 2)
    if rank_col == self.replication - 1:
        stages = (self.size // self.replication) - (self.replication - 1) * stages

    # row_data_send = [torch.cuda.FloatTensor(device=self.device)]*len(col_procs)
    
    row_data_send = []
    row_indices_recv = self.row_indices_recv
    row_data_recv = [torch.cuda.FloatTensor(device=self.device)]*len(col_procs)
    row_indices_send = self.row_indices_send
    
    start = time.time()
    for i in range(stages):
        q = (rank_col * (self.size // (self.replication ** 2)) + i) * self.replication + rank_col
        q_c = q // self.replication
        am_partid = rank_col * (self.size // self.replication ** 2) + i
        unique_cols = self.row_indices_send[q]
        if rank == q:
            # send_ops = []
            for j in col_procs:
                if rank != j:
                    # start = time.time()
                    rows_send = inputs[row_indices_recv[j].long(), :].clone()
                    # stop_time(self, "gather_row_data", start, barrier=False)
                    # start = time.time()
                    # dist.send(rows_send, dst=j, group=self.col_groups[rank_col])
                    row_data_send.append(rows_send)
                    # send_ops.append(dist.P2POp(dist.isend, rows_send, j, group=self.col_groups[rank_col]))
                    # stop_time(self, "send_ops", start, barrier=False)
                else:
                    row_data_send.append(torch.cuda.FloatTensor(device=self.device))
                    # inputs_mul = inputs.clone()
                    # spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                    #             ampbyp[i].values(), ampbyp[i].size(0),
                    #             ampbyp[i].size(1), inputs_mul, z_loc)

        else: # receiving data from q
            row_data_recv[q_c] = torch.cuda.FloatTensor(device=self.device).resize_((unique_cols.size(0), inputs.size(1))).fill_(0)

    if len(row_data_send) == 0: # not q for either stage
        row_data_send = [torch.cuda.FloatTensor(device=self.device)]*len(col_procs) 
    stop_time(self, "gather_row_data", start, barrier=False)

    start = time.time()
    dist.all_to_all(row_data_recv, row_data_send, group=self.col_groups[rank_col])
    stop_time(self, "a2a", start, barrier=False)

    start = time.time()
    for i in range(len(col_procs)):
        if row_data_recv[i].size()[0] != 0:
            #print(col_procs[i])
            inputs_mul = torch.cuda.FloatTensor( device = self.device).resize_(ampbyp[i].size(1), inputs.size(1)).fill_(0)
            inputs_mul[row_indices_send[col_procs[i]]] =  row_data_recv[i]

            spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                                ampbyp[i].values(), ampbyp[i].size(0),
                                ampbyp[i].size(1), inputs_mul, z_loc)
        elif rank == col_procs[i] and i >= rank_col * (self.size // (self.replication ** 2)) and i < rank_col * (self.size // (self.replication ** 2)) + stages:
            #print(col_procs[i])
            inputs_mul = inputs.clone()
            spmm_gpu(ampbyp[i].indices()[0].int(), ampbyp[i].indices()[1].int(),
                                ampbyp[i].values(), ampbyp[i].size(0),
                                ampbyp[i].size(1), inputs_mul, z_loc)
    stop_time(self, "spmm_gpu", start, barrier=False)

    z_loc = z_loc.contiguous()
    start = time.time()
    dist.all_reduce(z_loc, op=dist.reduce_op.SUM, group=self.row_groups[rank_c])
    stop_time(self, "reduce", start, barrier=False)
    """

    return z_loc
    





def summa_sparse(self, adj_matrix, inputs, height, middim, width):

    height_per_proc = height // self.proc_row
    width_per_proc  = width // self.proc_col

    middim_per_proc = middim // self.proc_col

    if self.rank_row == self.proc_row - 1:
        height_per_proc = height - height_per_proc * (self.proc_row - 1)

    if self.rank_col == self.proc_col - 1:
        width_per_proc = width - width_per_proc * (self.proc_col - 1)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=self.device).fill_(0)

    for k in range(self.proc_col):

        row_src_rank = k + self.proc_col * self.rank_row
        col_src_rank = k * self.proc_col + self.rank_col

        if k == self.proc_col - 1:
            middim_per_proc = middim - middim_per_proc * (self.proc_col - 1)

        if row_src_rank == self.rank:
            acol_indices_len = torch.cuda.LongTensor(
                                            [adj_matrix.indices().contiguous()[0].size(0)], 
                                            device=self.device)
            acol_values_len = torch.cuda.LongTensor([adj_matrix.values().contiguous().size(0)],
                                                    device=self.device)
        else:
            acol_indices_len = torch.cuda.LongTensor([0], device=self.device)
            acol_values_len = torch.cuda.LongTensor([0], device=self.device)

        dist.broadcast(acol_indices_len, row_src_rank, self.row_groups[self.rank_row])

        acol_indices_len = acol_indices_len.item() # nnz
        acol_values_len = acol_indices_len

        if row_src_rank == self.rank:
            acol_indices = adj_matrix.indices().contiguous().long()
            acol_values = adj_matrix.values().contiguous().float()
        else:
            acol_indices = torch.cuda.LongTensor(2, acol_indices_len, device=self.device).fill_(0)
            acol_values = torch.cuda.FloatTensor(acol_values_len, device=self.device).fill_(0)
        
        acol = torch.cat((acol_indices.float(), acol_values.unsqueeze(0)), dim=0).contiguous()

        dist.broadcast(acol, row_src_rank, self.row_groups[self.rank_row])

        acol_indices = acol[:2].long()
        acol_values = acol[2].squeeze(0)

        if row_src_rank == self.rank:
            acol = adj_matrix
        else:
            acol = sparse_coo_tensor_gpu(acol_indices, acol_values, 
                                            torch.Size([height_per_proc, middim_per_proc]))

        if col_src_rank == self.rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=self.device)

        brow = brow.contiguous()

        dist.broadcast(brow, col_src_rank, self.col_groups[self.rank_col])

        spmm_gpu(acol_indices[0].int(), acol_indices[1].int(), acol_values, 
                        height_per_proc, middim_per_proc, brow, z_loc)

    return z_loc

def summa_loc(self, mata, matb, height, middim, width):

    height_per_proc = height // self.proc_row
    width_per_proc  = width // self.proc_col
    middim_per_proc = middim // self.proc_row

    if self.rank_row == self.proc_row - 1:
        height_per_proc = height - height_per_proc * (self.proc_row - 1)

    width_per_proc = matb[self.rank].size(1)

    acol_tens = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=self.device)
    brow_tens = torch.FloatTensor(middim_per_proc, width_per_proc)

    acol = acol_tens
    brow = brow_tens

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=self.device).fill_(0)

    for k in range(self.proc_col):

        row_src_rank = k + self.proc_col * self.rank_row
        col_src_rank = k * self.proc_col + self.rank_col

        if k == self.proc_col - 1:
            middim_per_proc -= self.proc_col * middim_per_proc - middim

        if row_src_rank == self.rank:
            acol = mata
        else:
            acol = acol_tens
            acol = torch.cuda.FloatTensor(height_per_proc, matb[col_src_rank].size(0), 
                                            device=self.device)
        
        acol = acol.contiguous()
        dist.broadcast(acol, row_src_rank, self.row_groups[self.rank_row])

        brow = matb[col_src_rank]

        z_loc += torch.mm(acol, brow)

    return z_loc

def summa(self, adj_matrix, inputs, height, middim, width):

    height_per_proc = height // self.proc_row
    width_per_proc  = width // self.proc_col
    middim_per_proc = middim // self.proc_row

    if self.rank_row == self.proc_row - 1:
        height_per_proc = height - height_per_proc * (self.proc_row - 1)

    if self.rank_col == self.proc_col - 1:
        width_per_proc = width - width_per_proc * (self.proc_col - 1)

    acol_tens = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=self.device)
    brow_tens = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=self.device)

    acol = acol_tens
    brow = brow_tens

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=self.device).fill_(0)

    for k in range(self.proc_col):

        row_src_rank = k + self.proc_col * self.rank_row
        col_src_rank = k * self.proc_col + self.rank_col

        if k == self.proc_col - 1:
            middim_per_proc = middim - middim_per_proc * (self.proc_col - 1)
            acol_tens = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=self.device)
            brow_tens = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=self.device)

        if row_src_rank == self.rank:
            acol = adj_matrix
        else:
            acol = acol_tens
        
        acol = acol.contiguous()
        dist.broadcast(acol, row_src_rank, self.row_groups[self.rank_row])

        if col_src_rank == self.rank:
            brow = inputs
        else:
            brow = brow_tens

        brow = brow.contiguous()
        dist.broadcast(brow, col_src_rank, self.col_groups[self.rank_col])

        z_loc += torch.mm(acol.float(), brow)

    return z_loc

def transpose(self, mat, height, width, height_c=None, width_c=None):
    if self.partitioning == Partitioning.TWOD:
        return transpose_TWOD(self, mat, height, width) 
    elif self.partitioning == Partitioning.THREED:
        return transpose_THREED(self, mat, height, width, height_c, width_c) 

def transpose_TWOD(self, mat, height, width):
    rank = self.rank_row * self.proc_col + self.rank_col
    rank_t  = self.rank_col * self.proc_row + self.rank_row

    if rank == rank_t:
        return mat.t()

    height_recv = width // self.proc_row
    width_recv  = height // self.proc_col

    if self.rank_row == self.proc_row - 1:
        height_recv = width - height_recv * (self.proc_row - 1)

    if self.rank_col == self.proc_col - 1:
        width_recv = height - width_recv * (self.proc_col - 1)

    mat_recv = torch.cuda.FloatTensor(height_recv, width_recv, device=self.device)

    mat_recvs = [mat.t().contiguous(), mat_recv]

    if self.rank < rank_t:
        dist.broadcast(mat_recvs[0], src=self.rank, group=self.transpose_group)
        dist.broadcast(mat_recvs[1], src=rank_t, group=self.transpose_group)
    else:
        dist.broadcast(mat_recvs[1], src=rank_t, group=self.transpose_group)
        dist.broadcast(mat_recvs[0], src=self.rank, group=self.transpose_group)

    return mat_recvs[1]

def transpose_THREED(self, mat, height, width, height_c, width_c):
    rank_t = self.rank_col * self.proc_col * self.proc_c + self.rank_row * self.proc_c + self.rank_c 
    no_occur_val = 42.1234

    width_per_proc_c = width_c // self.proc_c
    mat_c_recv_width = width_c - width_per_proc_c * (self.proc_c - 1)

    if mat.size(1) != mat_c_recv_width:
        pad_col = mat_c_recv_width - mat.size(1)
        mat = torch.cat(
                    (mat, torch.cuda.FloatTensor(mat.size(0), pad_col, device=self.device).fill_(no_occur_val)),
                    dim=1) 

    mat_c_recv = []
    for i in range(self.proc_c):
        mat_c_recv.append(torch.cuda.FloatTensor(mat.size(), device=self.device))

    dist.all_gather(mat_c_recv, mat, group=self.c_groups[int(self.rank // self.proc_c)])
    for i in range(self.proc_c):
        mat_c_recv[i] = mat_c_recv[i].t_()
        mat_c_recv[i] = mat_c_recv[i][(mat_c_recv[i][:, 0] != no_occur_val).nonzero().squeeze(1)]
        mat_c_recv[i] = mat_c_recv[i].t_()

    mat = torch.cat(mat_c_recv, dim=1)
    mat.t_()

    if self.rank_col == self.proc_row - 1:
        width_recv = height - (height // self.proc_row) * (self.proc_row - 1)
    else:
        width_recv = height // self.proc_row

    if self.rank_row == self.proc_col - 1:
        height_recv = width - (width // self.proc_col) * (self.proc_col - 1)
    else:
        height_recv = width // self.proc_col

    if self.rank_row == self.rank_col:
        mat_recv = mat
    else:
        mat_recv = torch.cuda.FloatTensor(height_recv, width_recv, device=self.device)
        mat_recvs = [mat.contiguous(), mat_recv]

        if self.rank < rank_t:
            dist.broadcast(mat_recvs[0], src=self.rank, group=self.transpose_group)
            dist.broadcast(mat_recvs[1], src=rank_t, group=self.transpose_group)
        else:
            dist.broadcast(mat_recvs[1], src=rank_t, group=self.transpose_group)
            dist.broadcast(mat_recvs[0], src=self.rank, group=self.transpose_group)
         
    chunk_sizes_col = []
    chunk_len = width_recv // self.proc_c
    for i in range(self.proc_c):
        if i == self.proc_c - 1:
            chunk_sizes_col.append(width_recv - chunk_len * (self.proc_c - 1))
        else:
            chunk_sizes_col.append(chunk_len)
    mat_split = torch.split(mat_recv, chunk_sizes_col, dim=1)

    return mat_split[self.rank_c]

def outer_product(mata, matb, group):
    matc = torch.mm(mata, matb)
    dist.all_reduce(matc, op=dist.reduce_op.SUM, group=group)

    return matc

def split3dspmm_sparse(self, adj_matrix, inputs, height, middim, width):

    height_per_proc = height // self.proc_row
    width_per_proc  = width // self.proc_col
    middim_per_proc = middim // (self.proc_col * self.proc_c)

    if self.rank_row == self.proc_row - 1:
        height_per_proc = height - height_per_proc * (self.proc_row - 1)

    if self.rank_col == self.proc_col - 1:
        width_per_proc = width - width_per_proc * (self.proc_col - 1)

    width_per_proc_c = inputs.size(1) // self.proc_c
    if self.rank_c == self.proc_c - 1:
        width_per_proc_c = inputs.size(1) - width_per_proc_c * (self.proc_c - 1)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=self.device).fill_(0)

    chunk_sizes_col = []
    chunk_len = inputs.size(1) // self.proc_c
    for i in range(self.proc_c):
        if i == self.proc_c - 1:
            chunk_sizes_col.append(inputs.size(1) - chunk_len * (self.proc_c - 1))
        else:
            chunk_sizes_col.append(chunk_len)

    for k in range(self.proc_col):

        row_src_rank = self.rank_row * (self.proc_col * self.proc_c) + self.rank_c + k * self.proc_c
        col_src_rank = self.rank_col * self.proc_row + self.rank_c + k * self.proc_c * self.proc_row

        middim_per_col = middim // self.proc_col
        if k == self.proc_col - 1:
            middim_per_col = middim - middim_per_col * (self.proc_col - 1)
        
        middim_per_proc = middim_per_col // self.proc_c
        if self.rank_c == self.proc_c - 1:
            middim_per_proc = middim_per_col - middim_per_proc * (self.proc_c - 1)

        if row_src_rank == self.rank:
            acol_indices_len = torch.cuda.LongTensor(
                                            [adj_matrix.indices().contiguous()[0].size(0)], 
                                            device=self.device)
            acol_values_len = torch.cuda.LongTensor([adj_matrix.values().contiguous().size(0)],
                                                    device=self.device)
        else:
            acol_indices_len = torch.cuda.LongTensor([0], device=self.device)
            acol_values_len = torch.cuda.LongTensor([0], device=self.device)

        dist.broadcast(acol_indices_len, row_src_rank, self.row_groups[self.rank_row][self.rank_c])

        acol_indices_len = acol_indices_len.item() # nnz
        acol_values_len = acol_indices_len

        if row_src_rank == self.rank:
            acol_indices = adj_matrix.indices().contiguous().long()
            acol_values = adj_matrix.values().contiguous().float()
        else:
            acol_indices = torch.cuda.LongTensor(2, acol_indices_len, device=self.device).fill_(0)
            acol_values = torch.cuda.FloatTensor(acol_values_len, device=self.device).fill_(0)

        acol = torch.cat((acol_indices.float(), acol_values.unsqueeze(0)), dim=0)

        dist.broadcast(acol.contiguous(), row_src_rank, self.row_groups[self.rank_row][self.rank_c])

        acol_indices = acol[:2].long()
        acol_values = acol[2].squeeze(0)

        if row_src_rank == self.rank:
            acol = adj_matrix
        else:
            acol = sparse_coo_tensor_gpu(acol_indices, acol_values, 
                                            torch.Size([height_per_proc, middim_per_proc]))

        if col_src_rank == self.rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=self.device)

        brow = brow.contiguous()
        dist.broadcast(brow, col_src_rank, self.col_groups[self.rank_col][self.rank_c])

        spmm_gpu(acol_indices[0].int(), acol_indices[1].int(), acol_values, 
                        height_per_proc, middim_per_proc, brow, z_loc)

    z_loc = z_loc.contiguous()

    dist.all_reduce(z_loc, group=self.c_groups[int(self.rank // self.proc_c)])
    z_loc = torch.split(z_loc, chunk_sizes_col, dim=1)
    z_loc = z_loc[self.rank_c].contiguous()

    return z_loc, chunk_sizes_col

def split3dspmm_loc(self, mata, matb, height, middim, width):

    height_per_proc = height // self.proc_row
    width_per_proc  = width // self.proc_col
    middim_per_proc = middim // (self.proc_col * self.proc_c)

    if self.rank_row == self.proc_row - 1:
        height_per_proc = height - height_per_proc * (self.proc_row - 1)

    width_per_proc = matb[self.rank].size(1)

    height_per_proc_c = mata.size(0) // self.proc_c
    if self.rank_c == self.proc_c - 1:
        height_per_proc_c = mata.size(0) - height_per_proc_c * (self.proc_c - 1)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=self.device).fill_(0)

    chunk_sizes_row = []
    chunk_len = mata.size(0) // self.proc_c
    for i in range(self.proc_c):
        if i == self.proc_c - 1:
            chunk_sizes_row.append(mata.size(0) - chunk_len * (self.proc_c - 1))
        else:
            chunk_sizes_row.append(chunk_len)

    for k in range(self.proc_col):

        row_src_rank = self.rank_row * (self.proc_col * self.proc_c) + self.rank_c + k * self.proc_c
        col_src_rank = self.rank_col * self.proc_row + self.rank_c + k * self.proc_c * self.proc_row

        middim_per_col = middim // self.proc_col
        if k == self.proc_col - 1:
            middim_per_col = middim - middim_per_col * (self.proc_col - 1)
        
        middim_per_proc = middim_per_col // self.proc_c
        if self.rank_c == self.proc_c - 1:
            middim_per_proc = middim_per_col - middim_per_proc * (self.proc_c - 1)

        if row_src_rank == self.rank:
            acol = mata
        else:
            acol = torch.cuda.FloatTensor(height_per_proc, matb[col_src_rank].size(0), 
                                            device=self.device)
        
        dist.broadcast(acol.contiguous(), row_src_rank, self.row_groups[self.rank_row][self.rank_c])

        brow = matb[col_src_rank]

        z_tmp = torch.mm(acol, brow)

        dist.all_reduce(z_tmp, group=self.c_groups[int(self.rank // self.proc_c)])

        z_loc += z_tmp

    z_tmp = torch.split(z_loc, chunk_sizes_row, dim=0)
    z_loc = z_tmp[self.rank_c].clone()
    return z_loc

def split3dspmm_dense(self, adj_matrix, inputs, height, middim, width):

    height_per_proc = height // self.proc_row
    width_per_proc  = width // self.proc_col
    middim_per_proc = middim // (self.proc_col * self.proc_c)

    if self.rank_row == self.proc_row - 1:
        height_per_proc = height - height_per_proc * (self.proc_row - 1)

    if self.rank_col == self.proc_col - 1:
        width_per_proc = width - width_per_proc * (self.proc_col - 1)

    height_per_proc_c = adj_matrix.size(0) // self.proc_c
    if self.rank_c == self.proc_c - 1:
        height_per_proc_c = adj_matrix.size(0) - height_per_proc_c * (self.proc_c - 1)

    z_loc = torch.cuda.FloatTensor(height_per_proc, width_per_proc, device=self.device).fill_(0)

    chunk_sizes_row = []
    chunk_len = adj_matrix.size(0) // self.proc_c
    for i in range(self.proc_c):
        if i == self.proc_c - 1:
            chunk_sizes_row.append(adj_matrix.size(0) - chunk_len * (self.proc_c - 1))
        else:
            chunk_sizes_row.append(chunk_len)

    for k in range(self.proc_col):

        row_src_rank = self.rank_row * (self.proc_col * self.proc_c) + self.rank_c + k * self.proc_c
        col_src_rank = self.rank_col * self.proc_row + self.rank_c + k * self.proc_c * self.proc_row
        
        middim_per_col = middim // self.proc_col
        if k == self.proc_col - 1:
            middim_per_col = middim - middim_per_col * (self.proc_col - 1)
        
        middim_per_proc = middim_per_col // self.proc_c
        if self.rank_c == self.proc_c - 1:
            middim_per_proc = middim_per_col - middim_per_proc * (self.proc_c - 1)

        if row_src_rank == self.rank:
            acol = adj_matrix
        else:
            acol = torch.cuda.FloatTensor(height_per_proc, middim_per_proc, device=self.device).fill_(0)
        
        acol = acol.contiguous()
        dist.broadcast(acol, row_src_rank, self.row_groups[self.rank_row][self.rank_c])

        if col_src_rank == self.rank:
            brow = inputs
        else:
            brow = torch.cuda.FloatTensor(middim_per_proc, width_per_proc, device=self.device).fill_(0)

        brow = brow.contiguous()
        dist.broadcast(brow, col_src_rank, self.col_groups[self.rank_col][self.rank_c])

        z_loc += torch.mm(acol, brow)

    dist.all_reduce(z_loc, group=self.c_groups[int(self.rank // self.proc_c)])
    z_loc = torch.split(z_loc, chunk_sizes_row, dim=0)
    z_loc = z_loc[self.rank_c].contiguous()

    return z_loc

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, partitioning, device):
        super(GCNConv, self).__init__()

        weight_nonleaf = torch.rand(in_feats, out_feats, requires_grad=True)
        weight_nonleaf = weight_nonleaf.to(device)
        weight_nonleaf.retain_grad()
        self.weight = nn.Parameter(weight_nonleaf)
        self.partitioning = partitioning

    def forward(self, gcn, graph, inputs, ampbyp=None):
        if self.partitioning == Partitioning.ONED:
            return GCNFuncONED.apply(gcn, graph, ampbyp, inputs, self.weight)
        elif self.partitioning == Partitioning.ONE5D:
            return GCNFuncONE5D.apply(gcn, graph, ampbyp, inputs, self.weight)
        elif self.partitioning == Partitioning.TWOD:
            return GCNFuncTWOD.apply(gcn, graph, inputs, self.weight)
        elif self.partitioning == Partitioning.THREED:
            return GCNFuncTHREED.apply(gcn, graph, inputs, self.weight)
        

class GCNFuncONED(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, graph, ampbyp, inputs, weight):
        # inputs: H
        # graph: A
        # weight: W

        z = broad_func_oned(self, graph, ampbyp, inputs)
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
        ag = broad_func_oned(self, graph, ampbyp, grad_output)

        grad_input = ag.mm(weight.t())
        grad_weight = outer_product(inputs.t(), ag, self.group)

        return None, None, None, grad_input, grad_weight

class GCNFuncONE5D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, graph, ampbyp, inputs, weight):
        # inputs: H
        # graph: A
        # weight: W

        z = broad_func_one5d(self, graph, ampbyp, inputs)
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
        ag = broad_func_one5d(self, graph, ampbyp, grad_output)

        grad_input = ag.mm(weight.t())
        grad_weight = outer_product(inputs.t(), ag, self.group)

        # del ag

        return None, None, None, grad_input, grad_weight

class GCNFuncTWOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, graph, inputs, weight):
        # inputs: H
        # graph: A
        # weight: W

        graph_t = graph # Only true for undirected graphs

        z = summa_sparse(self, graph_t, inputs, self.node_count, self.node_count, weight.size(0))

        chunk_sizes_row = []
        chunk_sizes_col = []
        weight_per_row = weight.size(0) // self.proc_row
        weight_per_col = weight.size(1) // self.proc_col
        for i in range(self.proc_row):
            if i == self.proc_row - 1:
                chunk_sizes_row.append(weight.size(0) - weight_per_row * (self.proc_row - 1))
            else:
                chunk_sizes_row.append(weight_per_row)

        for i in range(self.proc_col):
            if i == self.proc_col - 1:
                chunk_sizes_col.append(weight.size(1) - weight_per_col * (self.proc_col - 1))
            else:
                chunk_sizes_col.append(weight_per_col)

        weight_rows = torch.split(weight, chunk_sizes_row, dim=0)
        weight_parts = []
        for i in weight_rows:
            weight_cols = torch.split(i, chunk_sizes_col, dim=1)
            weight_parts.extend(weight_cols)

        z = summa_loc(self, z, weight_parts, self.node_count, weight.size(0), weight.size(1))

        ctx.save_for_backward(inputs, weight)
        ctx.graph = graph
        ctx.self = self

        return z

    @staticmethod
    def backward(ctx, grad_output):
        graph = ctx.graph
        inputs, weight = ctx.saved_tensors
        self = ctx.self

        # Assumes graph is undirected and A = A^T
        ag = summa_sparse(self, graph, grad_output, self.node_count, self.node_count, weight.t().size(0))

        chunk_sizes_row = []
        chunk_sizes_col = []
        weight_per_row = weight.t().size(0) // self.proc_row
        weight_per_col = weight.t().size(1) // self.proc_col

        for i in range(self.proc_row):
            if i == self.proc_row - 1:
                chunk_sizes_row.append(weight.t().size(0) - weight_per_row * (self.proc_row - 1))
            else:
                chunk_sizes_row.append(weight_per_row)

        for i in range(self.proc_col):
            if i == self.proc_col - 1:
                chunk_sizes_col.append(weight.t().size(1) - weight_per_col * (self.proc_col - 1))
            else:
                chunk_sizes_col.append(weight_per_col)

        weight_rows = torch.split(weight.t(), chunk_sizes_row, dim=0)

        weight_parts = []
        for i in weight_rows:
            weight_cols = torch.split(i, chunk_sizes_col, dim=1)
            weight_parts.extend(weight_cols)

        grad_input = summa_loc(self, ag, weight_parts, self.node_count, weight.t().size(0), weight.t().size(1))

        # Second backprop equation (reuses the A * G^l computation)
        inputs_t = transpose(self, inputs, self.node_count, weight.size(0))

        grad_weight = summa(self, inputs_t, ag, weight.size(0), self.node_count, weight.size(1))

        # Collect grad_weight's across processes
        grad_weight_recv = []
        max_row_chunk = max(chunk_sizes_col) # transpose
        max_col_chunk = max(chunk_sizes_row)
        for i in range(self.size):
            grad_weight_recv.append(torch.cuda.FloatTensor(
                                                max_row_chunk,
                                                max_col_chunk,
                                                device=self.device))

        pad_row = max_row_chunk - grad_weight.size(0)
        pad_col = max_col_chunk - grad_weight.size(1)

        # TODO: make this part less hacky
        no_occur_val = 42.1234
        grad_weight = torch.cat((grad_weight, 
                torch.cuda.FloatTensor(pad_row, grad_weight.size(1), device=self.device).fill_(no_occur_val)), 
                        dim=0) 
        grad_weight = torch.cat((grad_weight, 
                torch.cuda.FloatTensor(grad_weight.size(0), pad_col, device=self.device).fill_(no_occur_val)), 
                        dim=1) 

        dist.all_gather(grad_weight_recv, grad_weight)

        for i in range(len(grad_weight_recv)):
            grad_weight_recv[i] = grad_weight_recv[i][(grad_weight_recv[i][:, 0] != no_occur_val)
                                                                .nonzero().squeeze(1)]

            grad_weight_recv_t = grad_weight_recv[i].t()
            grad_weight_recv_t = grad_weight_recv_t[(grad_weight_recv_t[:, 0] != no_occur_val)
                                                                .nonzero().squeeze(1)]

            grad_weight_recv[i] = grad_weight_recv_t.t()
        
        grad_weight_fin = torch.cuda.FloatTensor(device=self.device)
        for i in range(self.proc_row):
            grad_weight_row = torch.cuda.FloatTensor(device=self.device)
            for j in range(self.proc_col):
                rank_wt = i * self.proc_row + j
                grad_weight_row = torch.cat((grad_weight_row, grad_weight_recv[rank_wt]), dim=1)
            grad_weight_fin = torch.cat((grad_weight_fin, grad_weight_row), dim=0)

        return None, None, grad_input, grad_weight_fin

class GCNFuncTHREED(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, graph, inputs, weight):
        # inputs: H
        # graph: A
        # weight: W

        graph_t = graph # Only true for undirected graphs

        z, chunk_sizes_loc = split3dspmm_sparse(self, graph_t, inputs, self.node_count, 
                                                        self.node_count, weight.size(0))

        chunk_sizes_loc_tens = torch.cuda.LongTensor(chunk_sizes_loc)
        chunk_sizes = []
        for i in range(self.proc_col):
            chunk_sizes.append(torch.cuda.LongTensor(chunk_sizes_loc_tens.size()))
        dist.all_gather(chunk_sizes, chunk_sizes_loc_tens, group=self.row_groups[self.rank_row][self.rank_c])
        chunk_sizes = torch.cat(chunk_sizes).tolist()

        chunk_sizes_row = []
        chunk_sizes_col = []
        weight_per_row = weight.size(0) // (self.proc_row * self.proc_c)
        weight_per_col = weight.size(1) // self.proc_col
        chunk_sizes_row = chunk_sizes

        for i in range(self.proc_col):
            if i == self.proc_col - 1:
                chunk_sizes_col.append(weight.size(1) - weight_per_col * (self.proc_col - 1))
            else:
                chunk_sizes_col.append(weight_per_col)

        weight_rows = torch.split(weight, chunk_sizes_row, dim=0)
        weight_parts_tmp = []
        for i in weight_rows:
            weight_cols = torch.split(i, chunk_sizes_col, dim=1)
            weight_parts_tmp.extend(weight_cols)

        weight_parts = [None] * self.size
        for i in range(self.proc_row * self.proc_c):
            for j in range(self.proc_col):
                rank_old = i * self.proc_col + j

                rank_new_row = i // self.proc_c
                rank_new_c = i % self.proc_c
                rank_new = rank_new_row * self.proc_col * self.proc_c + j * self.proc_c + rank_new_c
                
                if j == self.rank_col:
                    weight_parts[rank_new] = weight_parts_tmp[rank_old]
                else:
                    weight_parts[rank_new] = None

        z = split3dspmm_loc(self, z, weight_parts, self.node_count, weight.size(0), weight.size(1))

        ctx.save_for_backward(inputs, weight)
        ctx.graph = graph
        ctx.self = self

        return z

    @staticmethod
    def backward(ctx, grad_output):
        graph = ctx.graph
        inputs, weight = ctx.saved_tensors
        self = ctx.self

        # First backprop equation
        ag, chunk_sizes_loc = split3dspmm_sparse(self, graph, grad_output, self.node_count, 
                                                    self.node_count, weight.t().size(0))

        chunk_sizes_loc_tens = torch.cuda.LongTensor(chunk_sizes_loc)
        chunk_sizes = []
        for i in range(self.proc_col):
            chunk_sizes.append(torch.cuda.LongTensor(chunk_sizes_loc_tens.size()))
        dist.all_gather(chunk_sizes, chunk_sizes_loc_tens, group=self.row_groups[self.rank_row][self.rank_c])
        chunk_sizes = torch.cat(chunk_sizes).tolist()

        chunk_sizes_row = []
        chunk_sizes_col = []
        weight_per_row = weight.t().size(0) // (self.proc_row * self.proc_c)
        weight_per_col = weight.t().size(1) // self.proc_col
        chunk_sizes_row = chunk_sizes

        for i in range(self.proc_col):
            if i == self.proc_col - 1:
                chunk_sizes_col.append(weight.t().size(1) - weight_per_col * (self.proc_col - 1))
            else:
                chunk_sizes_col.append(weight_per_col)
        weight_rows = torch.split(weight.t(), chunk_sizes_row, dim=0)

        weight_parts_tmp = []
        for i in weight_rows:
            weight_cols = torch.split(i, chunk_sizes_col, dim=1)
            weight_parts_tmp.extend(weight_cols)

        weight_parts = [None] * self.size
        for i in range(self.proc_row * self.proc_c):
            for j in range(self.proc_col):
                rank_old = i * self.proc_col + j

                rank_new_row = i // self.proc_c
                rank_new_c = i % self.proc_c
                rank_new = rank_new_row * self.proc_col * self.proc_c + j * self.proc_c + rank_new_c
                
                if self.rank_col == j:
                    weight_parts[rank_new] = weight_parts_tmp[rank_old]
                else:
                    weight_parts[rank_new] = None

        grad_input = split3dspmm_loc(self, ag, weight_parts, self.node_count, weight.t().size(0), weight.size(1))

        # Second backprop equation (reuses the A * G^l computation)
        # col_groups twice because of transpose
        height_c = self.node_count // self.proc_row
        if self.rank_row == self.proc_row - 1:
            height_c = self.node_count - height_c * (self.proc_row - 1)

        width_c = weight.size(1) // self.proc_col
        if self.rank_col == self.proc_col - 1:
            width_c = weight.size(1) - width_c * (self.proc_col - 1)

        ag_t = transpose(self, ag, self.node_count, weight.size(1), height_c, width_c)

        grad_weight = split3dspmm_dense(self, ag_t, inputs, weight.size(1), self.node_count, weight.size(0))
        
        grad_weight_recv = []
        max_row_chunk = max(chunk_sizes_row)
        max_col_chunk = max(chunk_sizes_col)
        for i in range(self.size):
            grad_weight_recv.append(torch.cuda.FloatTensor(
                                                max_row_chunk,
                                                max_col_chunk,
                                                device=self.device))

        pad_row = max_row_chunk - grad_weight.size(0)
        pad_col = max_col_chunk - grad_weight.size(1)

        # TODO: make this part less hacky
        no_occur_val = 42.1234
        grad_weight = torch.cat((grad_weight, 
                torch.cuda.FloatTensor(pad_row, grad_weight.size(1), device=self.device).fill_(no_occur_val)), 
                dim=0) 
        grad_weight = torch.cat((grad_weight, 
                torch.cuda.FloatTensor(grad_weight.size(0), pad_col, device=self.device).fill_(no_occur_val)), 
                dim=1) 

        dist.all_gather(grad_weight_recv, grad_weight)

        for i in range(len(grad_weight_recv)):
            grad_weight_recv[i] = grad_weight_recv[i][(grad_weight_recv[i][:, 0] != no_occur_val)
                                                                .nonzero().squeeze(1)]

            grad_weight_recv_t = grad_weight_recv[i].t()
            grad_weight_recv_t = grad_weight_recv_t[(grad_weight_recv_t[:, 0] != no_occur_val)
                                                                .nonzero().squeeze(1)]

            grad_weight_recv[i] = grad_weight_recv_t.t()
        
        grad_weight_fin = torch.cuda.FloatTensor(device=self.device)
        for i in range(self.proc_row):
            grad_weight_row = torch.cuda.FloatTensor(device=self.device)
            for j in range(self.proc_col):
                grad_weight_col = torch.cuda.FloatTensor(device=self.device)
                for k in range(self.proc_c):
                    rank_wt = i * self.proc_row * self.proc_c + j * self.proc_c + k
                    grad_weight_col = torch.cat((grad_weight_col, grad_weight_recv[rank_wt]), dim=0)
                grad_weight_row = torch.cat((grad_weight_row, grad_weight_col), dim=1)
            grad_weight_fin = torch.cat((grad_weight_fin, grad_weight_row), dim=0)

        grad_weight_fin = grad_weight_fin.t()

        return None, None, grad_input, grad_weight_fin
