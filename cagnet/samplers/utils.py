import math 
import torch 
import torch.distributed as dist 
import torch_sparse 
from collections import defaultdict

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, \
                                    compute_darts_select_gpu, throw_darts_select_gpu, \
                                    compute_darts1d_gpu, throw_darts1d_gpu, normalize_gpu, \
                                    shift_rowselect_gpu, shift_colselect_gpu, \
                                    scatterd_add_gpu, scatteri_add_gpu, rowselect_coo_gpu, \
                                    rowselect_csr_gpu, sparse_coo_tensor_gpu, spgemm_gpu, coogeam_gpu, \
                                    sparse_csr_tensor_gpu


timing = True

def start_time(timer):
    if timing:
        timer.record()

def stop_time(start_timer, stop_timer, barrier=False):
    if timing:
        stop_timer.record()
        torch.cuda.synchronize()
        time_taken = start_timer.elapsed_time(stop_timer)
        if barrier:
            dist.barrier()
        return time_taken
    else:
        return 0.0

def stop_time_add(start_timer, stop_timer, timing_dict, range_name, barrier=False):
    if timing_dict is not None:
        timing_dict[range_name].append(stop_time(start_timer, stop_timer, barrier))

def csr_allreduce(mat, left, right, rank, name=None, timing_dict=None):
    while left < right:
        group_size = right - left + 1
        mid_rank = (left + right) // 2

        if rank <= mid_rank:
            recv_rank = rank + (group_size // 2)
        else:
            recv_rank = rank - (group_size // 2)

        ops = [None, None]
        nnz_send = torch.cuda.IntTensor(1).fill_(mat._nnz())
        nnz_recv = torch.cuda.IntTensor(1)
        
        ops[0] = dist.P2POp(dist.isend, nnz_send, recv_rank)
        ops[1] = dist.P2POp(dist.irecv, nnz_recv, recv_rank)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait() 

        torch.cuda.synchronize()
        rows_recv = torch.cuda.IntTensor(mat.size(0) + 1)
        cols_recv = torch.cuda.IntTensor(nnz_recv.item())
        vals_recv = torch.cuda.FloatTensor(nnz_recv.item())

        ops = [None] * 6
        ops[0] = dist.P2POp(dist.isend, mat.crow_indices().int(), recv_rank, tag=0)
        ops[1] = dist.P2POp(dist.isend, mat.col_indices().int(), recv_rank, tag=1)
        ops[2] = dist.P2POp(dist.isend, mat.values(), recv_rank, tag=2)
        ops[3] = dist.P2POp(dist.irecv, rows_recv, recv_rank, tag=0)
        ops[4] = dist.P2POp(dist.irecv, cols_recv, recv_rank, tag=1)
        ops[5] = dist.P2POp(dist.irecv, vals_recv, recv_rank, tag=2)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait() 

        torch.cuda.synchronize()
        mat_recv = torch.sparse_csr_tensor(rows_recv, cols_recv, vals_recv, size=mat.size())
        mat = mat + mat_recv

        if rank <= mid_rank:
            right = mid_rank
        else:
            left = mid_rank + 1
    return mat

def dist_spgemm15D(mata, matb, replication, rank, size, row_groups, col_groups, name, timing_dict=None):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_time(start_timer)
    chunk_size = math.ceil(float(mata.size(1) / (size / replication)))
    matc = torch.sparse_coo_tensor(size=(mata.size(0), matb.size(1))).cuda()
    rank_c = rank // replication
    rank_col = rank % replication
    stages = size // (replication ** 2)
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-inst-{name}")

    for i in range(stages):
        start_time(start_timer)
        q = (rank_col * (size // (replication ** 2)) + i) * replication + rank_col

        matb_recv_nnz = torch.cuda.IntTensor([matb._nnz()])
        dist.broadcast(matb_recv_nnz, src=q, group=col_groups[rank_col])
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-bcast-nnz-{name}")

        start_time(start_timer)
        if q == rank:
            matb_recv_indices = matb._indices().clone()
            matb_recv_values = matb._values().clone()
        else:
            matb_recv_indices = torch.cuda.LongTensor(2, matb_recv_nnz.item())
            matb_recv_values = torch.cuda.DoubleTensor(matb_recv_nnz.item())
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-inst-recv-{name}")

        start_time(start_timer)
        dist.broadcast(matb_recv_indices, src=q, group=col_groups[rank_col])
        dist.broadcast(matb_recv_values, src=q, group=col_groups[rank_col])
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-bcast-data-{name}")

        start_time(start_timer)
        am_partid = rank_col * (size // replication ** 2) + i
        chunk_col_start = am_partid * chunk_size
        chunk_col_stop = min((am_partid + 1) * chunk_size, mata.size(1))
        chunk_col_size = chunk_col_stop - chunk_col_start
        chunk_col_mask = (mata._indices()[1, :] >= chunk_col_start) & (mata._indices()[1, :] < chunk_col_stop)

        mata_chunk_indices = mata._indices()[:, chunk_col_mask]
        mata_chunk_indices[1,:] -= chunk_col_start
        mata_chunk_values = mata._values()[chunk_col_mask]
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-local-{name}")

        start_time(start_timer)
        matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_chunk_indices, \
                                                    mata_chunk_values.double(), matb_recv_indices, \
                                                    matb_recv_values, mata.size(0), \
                                                    chunk_col_size, matb.size(1), coalesced=True)

        # print(f"before spgemm_gpu", flush=True)
        # print(f"mata_chunk_indices[0,:]: {mata_chunk_indices[0,:].int()}", flush=True)
        # print(f"mata_chunk_indices[0,:].size: {mata_chunk_indices[0,:].int().size()}", flush=True)
        # print(f"mata.size(0): {mata.size(0)}", flush=True)
        # mata_chunk_indices = mata_chunk_indices.int()
        # matb_recv_indices = matb_recv_indices.int()
        # print(f"matb_recv_indices.row: {matb_recv_indices[0,:]}", flush=True)
        # print(f"matb_recv_values: {matb_recv_values}", flush=True)
        # matc_outputs = spgemm_gpu(mata_chunk_indices[0,:],      # A_rows
        #                             mata_chunk_indices[1,:],    # A_cols
        #                             mata_chunk_values.float(),  # A_vals
        #                             matb_recv_indices[0,:],     # B_rows
        #                             matb_recv_indices[1,:],     # B_cols
        #                             matb_recv_values.float(),   # B_vals
        #                             mata.size(0),               # n
        #                             chunk_col_size,             # k
        #                             matb.size(1))               # m
        # print(f"after spgemm_gpu", flush=True)
        # print(f"matc_outputs: {matc_outputs}", flush=True)
        # mata_chunk_indices = mata_chunk_indices.long()
        # matb_recv_indices = matb_recv_indices.long()
        # matc_chunk_rows = matc_outputs[0].long()
        # matc_chunk_cols = matc_outputs[1].long()
        # matc_chunk_values = matc_outputs[2].double()

        # print(f"before matc_chunk_rows: {matc_chunk_rows}")
        # matc_chunk_counts = torch.diff(matc_chunk_rows)
        # matc_chunk_rows = torch.repeat_interleave(torch.arange(0, mata.size(0), device=torch.device("cuda:0")),
        #                                             matc_chunk_counts)
        # print(f"after matc_chunk_rows: {matc_chunk_rows}")
        # matc_chunk_indices = torch.stack((matc_chunk_rows, matc_chunk_cols))
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-local-spgemm-{name}")

        start_time(start_timer)
        # matc_chunk = torch.sparse_coo_tensor(matc_chunk_indices, matc_chunk_values, size=matc.size())
        matc_chunk = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, matc.size())
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-chunk-inst-{name}")

        start_time(start_timer)
        matc_chunk = matc_chunk.coalesce()
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-chunk-coalesce-{name}")

        start_time(start_timer)
        matc += matc_chunk
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-chunk-add-{name}")

    start_time(start_timer)
    matc = matc.coalesce()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-coalesce-{name}")

    # dist.all_reduce(matc, op=dist.reduce_op.SUM, group=row_groups[rank_c])
    # Implement sparse allreduce w/ all_gather and padding
    start_time(start_timer)
    matc_nnz = torch.cuda.IntTensor(1).fill_(matc._nnz())
    dist.all_reduce(matc_nnz, dist.ReduceOp.MAX, row_groups[rank_c])
    matc_nnz = matc_nnz.item()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-nnz-{name}")

    start_time(start_timer)
    matc_recv_indices = []
    matc_recv_values = []
    for i in range(replication):
        matc_recv_indices.append(torch.cuda.LongTensor(2, matc_nnz).fill_(0))
        matc_recv_values.append(torch.cuda.DoubleTensor(matc_nnz).fill_(0.0))

    matc_send_indices = torch.cat((matc._indices(), torch.cuda.LongTensor(2, matc_nnz - matc._nnz()).fill_(0)), 1)
    matc_send_values = torch.cat((matc._values(), torch.cuda.DoubleTensor(matc_nnz - matc._nnz()).fill_(0.0)))
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-padding-{name}")

    start_time(start_timer)
    dist.all_gather(matc_recv_indices, matc_send_indices, row_groups[rank_c])
    dist.all_gather(matc_recv_values, matc_send_values, row_groups[rank_c])
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-allgather-{name}")

    start_time(start_timer)
    matc_recv = []
    for i in range(replication):
        # matc_recv.append(torch.sparse_coo_tensor(matc_recv_indices[i], matc_recv_values[i], matc.size()))
        matc_recv.append(sparse_coo_tensor_gpu(matc_recv_indices[i], matc_recv_values[i], matc.size()))
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-reduce-{name}")

    start_time(start_timer)
    matc_recv = torch.stack(matc_recv)
    matc = torch.sparse.sum(matc_recv, dim=0)
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-{name}")
    start_time(start_timer)
    matc = matc.coalesce()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-coalesce-{name}")

    start_time(start_timer)
    nnz_mask = matc._values() > 0
    matc_nnz_indices = matc._indices()[:, nnz_mask]
    matc_nnz_values = matc._values()[nnz_mask]
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-unpad-{name}")

    return matc_nnz_indices, matc_nnz_values

def dist_saspgemm15D(mata, matb, replication, rank, size, row_groups, col_groups, \
                            name, nnz_row_masks, matb_recv_buff, timing_dict=None, alg=None):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_inner_timer = torch.cuda.Event(enable_timing=True)
    stop_inner_timer = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("nvtx-matc-inst")
    start_time(start_timer)
    chunk_size = math.ceil(float(mata.size(1) / (size / replication)))
    if True or mata.layout == torch.sparse_coo:
        matc = torch.sparse_coo_tensor(size=(mata.size(0), matb.size(1))).cuda()
    elif mata.layout == torch.sparse_csr:
        matc_crow_indices = torch.cuda.LongTensor(mata.size(0) + 1).fill_(0)
        matc = torch.sparse_csr_tensor(matc_crow_indices, \
                                            torch.cuda.LongTensor(0), \
                                            torch.cuda.FloatTensor(0),
                                            size=(mata.size(0), matb.size(1)))
    rank_c = rank // replication
    rank_col = rank % replication
    stages = size // (replication ** 2)
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-inst-{name}", barrier=True)
    torch.cuda.nvtx.range_pop() # matc-inst

    for i in range(stages):
        # start_time(start_timer)
        q = (rank_col * (size // (replication ** 2)) + i) * replication + rank_col

        # Extract chunk of mata for local SpGEMM
        torch.cuda.nvtx.range_push("nvtx-preproc-local")
        start_time(start_timer)
        am_partid = rank_col * (size // replication ** 2) + i
        chunk_col_start = am_partid * chunk_size
        chunk_col_stop = min((am_partid + 1) * chunk_size, mata.size(1))
        chunk_col_size = chunk_col_stop - chunk_col_start
        if mata.layout == torch.sparse_coo:
            chunk_col_mask = (mata._indices()[1, :] >= chunk_col_start) & \
                             (mata._indices()[1, :] < chunk_col_stop)
            mata_chunk_indices = mata._indices()[:, chunk_col_mask]
            mata_chunk_indices[1,:] -= chunk_col_start
            mata_chunk_values = mata._values()[chunk_col_mask]
        elif mata.layout == torch.sparse_csr:
            # Column selection
            chunk_col_mask = (mata.col_indices() >= chunk_col_start) & \
                             (mata.col_indices() < chunk_col_stop)
            mata_chunk_cols = mata.col_indices()[chunk_col_mask]
            mata_chunk_cols -= chunk_col_start
            mata_chunk_values = mata.values()[chunk_col_mask]

            # # torch 1.13 supports offsets-based reduction
            # mata_chunk_rowcount = torch.segment_reduce(chunk_col_mask.float(), \
            #                                                 "sum", \
            #                                                 offsets=mata.crow_indices())

            lengths = mata.crow_indices()[1:] - mata.crow_indices()[:-1]
            mata_chunk_rowcount = torch.segment_reduce(chunk_col_mask.float(), \
                                                            "sum", \
                                                            lengths=lengths)
            mata_chunk_crows = torch.cuda.LongTensor(mata.size(0) + 1)
            mata_chunk_crows[1:] = torch.cumsum(mata_chunk_rowcount, dim=0).long()
            mata_chunk_crows[0] = 0

        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-local-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # preproc-local

        # Determine number of nonzero columns in chunk of mata
        torch.cuda.nvtx.range_push("nvtx-unique")
        start_time(start_timer)
        if mata.layout == torch.sparse_coo:
            nnz_cols = torch.unique(mata_chunk_indices[1,:])
        elif mata.layout == torch.sparse_csr:
            nnz_cols = torch.unique(mata_chunk_cols)
        nnz_cols_count = torch.cuda.IntTensor(1).fill_(nnz_cols.size(0))
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-unique-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # unique
        # timing_dict[f"spgemm-nnzcount-{name}"].append(nnz_cols.size(0))

        # Gather nnz column counts on rank q
        torch.cuda.nvtx.range_push("nvtx-gather-nnzcounts")
        start_time(start_timer)
        if rank == q:
            nnz_cols_count_list = []
            for j in range(size // replication):
                nnz_cols_count_list.append(torch.cuda.IntTensor(1).fill_(0))
                recv_rank = rank_col + j * replication
            dist.gather(nnz_cols_count, nnz_cols_count_list, dst=q, group=col_groups[rank_col])
        else:
            dist.gather(nnz_cols_count, dst=q, group=col_groups[rank_col])
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-gather-nnzcounts-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # gather-nnzcounts

        # Rank q allocates recv buffer for nnz col ids
        torch.cuda.nvtx.range_push("nvtx-alloc-nnzbuff")
        start_time(start_timer)
        if rank == q:
            nnz_col_ids = []
            for nnz_count in nnz_cols_count_list:
                nnz_col_ids.append(torch.cuda.LongTensor(nnz_count.item()).fill_(0))
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-alloc-nnzbuff-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # alloc-nnzbuff
        
        # isend/irecv nnz col ids to rank q
        torch.cuda.nvtx.range_push("nvtx-send-colids")
        start_time(start_timer)
        if rank == q:
            recv_objs = []
            for j in range(size // replication):
                recv_rank = rank_col + j * replication
                if recv_rank != q:
                    recv_objs.append(dist.recv(nnz_col_ids[j], src=recv_rank))
                else:
                    nnz_col_ids[j] = nnz_cols
        else:
            dist.send(nnz_cols, dst=q)
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-send-colids-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # send-nnzcols

        # Send selected rows count 
        torch.cuda.nvtx.range_push("nvtx-send-rowdata")
        start_time(start_timer)
        nnz_count = 0
        if rank == q:
            nnz_cols_count_list = torch.cat(nnz_cols_count_list, dim=0)
            if matb.layout == torch.sparse_coo:
                start_time(start_inner_timer)
                rowselect_coo_gpu(nnz_col_ids, matb._indices()[0,:], nnz_row_masks, nnz_cols_count_list, \
                                        matb._indices().size(1), size // replication)
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-rowsel-{name}")
            recv_proc_count = ((size // replication) - 1)
            send_ops = [None] * 4 * recv_proc_count
            send_idx = 0

            # start_time(start_inner_timer)
            for j in range(size // replication):
                recv_rank = rank_col + j * replication
                nnz_row_mask = nnz_row_masks[(j * matb._nnz()):((j + 1) * matb._nnz())]
                
                if matb.layout == torch.sparse_csr:
                    start_time(start_inner_timer)
                    rowselect_csr_gpu(nnz_col_ids[j], matb.crow_indices(), nnz_row_mask, \
                                            nnz_col_ids[j].size(0), matb._nnz())
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-rowsel-{name}")

                    start_time(start_inner_timer)
                    row_lengths = matb.crow_indices()[1:] - matb.crow_indices()[:-1]
                    if nnz_col_ids[j].size(0) > 0:
                        matb_send_lengths = row_lengths[nnz_col_ids[j]]
                    else:
                        matb_send_lengths = torch.cuda.LongTensor(0)

                    matb_send_cols = matb.col_indices()[nnz_row_mask]
                    matb_send_values = matb.values()[nnz_row_mask]
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-misc-{name}")

                    start_time(start_inner_timer)
                    if recv_rank != q:
                        selected_rows_count = torch.cuda.IntTensor(1).fill_(matb_send_cols.size(0))
                        nnz_count += 2 * matb_send_cols.size(0) + nnz_col_ids[j].size(0)
                        # dist.send(selected_rows_count, tag=0, dst=recv_rank)
                        # # dist.send(matb_send_crows, tag=1, dst=recv_rank)
                        # dist.send(matb_send_lengths, tag=1, dst=recv_rank)
                        # dist.send(matb_send_cols, tag=2, dst=recv_rank)
                        # dist.send(matb_send_values, tag=3, dst=recv_rank)

                        send_ops[send_idx] = \
                                    dist.P2POp(dist.isend, selected_rows_count, recv_rank, tag=0)
                        send_ops[send_idx + 1] = \
                                    dist.P2POp(dist.isend, matb_send_lengths.int(), recv_rank, tag=1)
                        send_ops[send_idx + 2] = \
                                    dist.P2POp(dist.isend, matb_send_cols.int(), recv_rank, tag=2)
                        send_ops[send_idx + 3] = \
                                    dist.P2POp(dist.isend, matb_send_values, recv_rank, tag=3)
                        send_idx += 4
                    else:
                        # matb_recv_crows = matb_send_crows.clone()
                        matb_recv_cols = matb_send_cols.clone()
                        matb_recv_values = matb_send_values.clone()
                        if not (name == "prob" and alg == "sage"):
                            matb_rows_sum = torch.cuda.LongTensor(matb.crow_indices().size(0) - 1).fill_(0)
                            if nnz_row_mask.any():
                                matb_rows_sum[nnz_col_ids[j]] = matb_send_lengths
                            matb_recv_crows = torch.cuda.LongTensor(matb.crow_indices().size(0))
                            matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dim=0)
                            matb_recv_crows[0] = 0
                        else:
                            matb_recv_rows = torch.repeat_interleave(nnz_col_ids[j], matb_send_lengths)
                            matb_recv_indices = torch.stack((matb_recv_rows, matb_recv_cols))
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-calls-{name}")
                elif matb.layout == torch.sparse_coo:
                    # rowselect_coo_gpu(nnz_col_ids[j], matb._indices()[0,:], nnz_row_mask, \
                    #                         nnz_col_ids[j].size(0), matb._indices().size(1))

                    start_time(start_inner_timer)
                    matb_send_indices = matb._indices()[:, nnz_row_mask]
                    matb_send_values = matb._values()[nnz_row_mask]
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-misc-{name}")

                    start_time(start_inner_timer)
                    if recv_rank != q:
                        selected_rows_count = torch.cuda.IntTensor(1).fill_(matb_send_indices.size(1))
                        nnz_count += 3 * matb_send_indices.size(1)
                        dist.send(selected_rows_count, tag=0, dst=recv_rank)
                        dist.send(matb_send_indices, tag=1, dst=recv_rank)
                        dist.send(matb_send_values, tag=2, dst=recv_rank)

                        # send_ops[send_idx] = dist.P2POp(dist.isend, selected_rows_count, recv_rank, tag=0)
                        # send_ops[send_idx + 1] = dist.P2POp(dist.isend, matb_send_indices, recv_rank, tag=1)
                        # send_ops[send_idx + 2] = dist.P2POp(dist.isend, matb_send_values, recv_rank, tag=2)
                        # send_idx += 3
                    else:
                        # matb_select_recv = matb_send.clone()
                        matb_recv_indices = matb_send_indices.clone()
                        matb_recv_values = matb_send_values.clone()
                    stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-send-calls-{name}")

            # if len(send_ops) > 0:
            if len(send_ops) > 0 and matb.layout == torch.sparse_csr:
                reqs = dist.batch_isend_irecv(send_ops)
                for req in reqs:
                    req.wait()
        else:
            if f"spgemm-send-rowsel-{name}" not in timing_dict:
                timing_dict[f"spgemm-send-rowsel-{name}"] = []
            
            if f"spgemm-send-misc-{name}" not in timing_dict:
                timing_dict[f"spgemm-send-misc-{name}"] = []

            if f"spgemm-send-calls-{name}" not in timing_dict:
                timing_dict[f"spgemm-send-calls-{name}"] = []

            start_time(start_inner_timer)
            selected_rows_count_recv = torch.cuda.IntTensor(1)
            dist.recv(selected_rows_count_recv, tag=0, src=q)

            if matb.layout == torch.sparse_csr:
                # matb_recv_crows = torch.cuda.LongTensor(chunk_col_size + 1)
                # dist.recv(matb_recv_crows, tag=1, src=q)

                # matb_recv_lengths = torch.cuda.LongTensor(nnz_cols.size(0))
                matb_recv_lengths = torch.cuda.IntTensor(nnz_cols.size(0))
                dist.recv(matb_recv_lengths, tag=1, src=q)
                matb_recv_lengths = matb_recv_lengths.long()

                # matb_recv_cols = torch.cuda.LongTensor(selected_rows_count_recv.item())
                matb_recv_cols = torch.cuda.IntTensor(selected_rows_count_recv.item())
                dist.recv(matb_recv_cols, tag=2, src=q)
                matb_recv_cols = matb_recv_cols.long()

                matb_recv_values = torch.cuda.DoubleTensor(selected_rows_count_recv.item())
                dist.recv(matb_recv_values, tag=3, src=q)

                if not (name == "prob" and alg == "sage"):
                    matb_rows_sum = torch.cuda.LongTensor(chunk_col_size).fill_(0)
                    if matb_recv_lengths.size(0) > 0:
                        matb_rows_sum[nnz_cols] = matb_recv_lengths
                    matb_recv_crows = torch.cuda.LongTensor(chunk_col_size + 1)
                    matb_recv_crows[1:] = torch.cumsum(matb_rows_sum, dim=0)
                    matb_recv_crows[0] = 0
                else:
                    matb_recv_rows = torch.repeat_interleave(nnz_cols, matb_recv_lengths)
                    matb_recv_indices = torch.stack((matb_recv_rows, matb_recv_cols))

                nnz_count += 2 * matb_recv_cols.size(0) + matb_recv_lengths.size(0)

            elif matb.layout == torch.sparse_coo:

                # matb_indices_recv = matb_recv_buff[:(3 * selected_rows_count_recv.item())].view(3, -1)
                matb_recv_indices = torch.cuda.LongTensor(2, selected_rows_count_recv.item())
                dist.recv(matb_recv_indices, tag=1, src=q)

                matb_recv_values = torch.cuda.DoubleTensor(selected_rows_count_recv.item())
                dist.recv(matb_recv_values, tag=2, src=q)

                # recv_ops = [dist.P2POp(dist.irecv, matb_recv_indices, q, tag=1), \
                #                 dist.P2POp(dist.irecv, matb_recv_values, q, tag=2)]

                # reqs = dist.batch_isend_irecv(recv_ops)
                # for req in reqs:
                #     req.wait()
                nnz_count += 3 * matb_recv_indices.size(1)
            stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-recv-calls-{name}")

        torch.cuda.synchronize()
        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-send-rowdata-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # send-rowcounts

        if f"spgemm-recv-calls-{name}" not in timing_dict:
            timing_dict[f"spgemm-recv-calls-{name}"] = []

        timing_dict[f"spgemm-send-rownnz-{name}"].append(nnz_count)

        # torch.cuda.nvtx.range_push("nvtx-spgemm-castrecv")
        # start_time(start_timer)
        # # print(f"matb_select_recv.dtype: {matb_select_recv.dtype}")
        # # matb_recv_indices = matb_select_recv[:2, :].long()
        # # # matb_recv_values = matb_select_recv[2, :].double()
        # # matb_recv_values = matb_select_recv[2, :]

        # # matb_recv_indices = matb_select_recv[:2, :].long()
        # # matb_recv_values = matb_select_recv[2, :].double()
        # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-castrecv-{name}", barrier=True)
        # torch.cuda.nvtx.range_pop() # spgemm-castrecv

        torch.cuda.nvtx.range_push("nvtx-local-spgemm")
        start_time(start_timer)
        if mata_chunk_values.size(0) > 0 and matb_recv_values.size(0) > 0:
            # Can skip local spgemm for row-selection for any alg and for SAGE's probability spgemm
            # if name == "prob" and alg == "sage":
            #     matc_chunk_crows = torch.cuda.LongTensor(mata.size(0)).fill_(0)
            #     matc_chunk = torch.sparse_csr_tensor(matb_recv_crows, matb_recv_cols, matb_recv_values.float(),
            #                                             size=(chunk_col_size, matb.size(1)))
            #     matc += matc_chunk
            if mata.layout == torch.sparse_csr:
                start_time(start_inner_timer)
                mata_recv = torch.sparse_csr_tensor(mata_chunk_crows, mata_chunk_cols, mata_chunk_values,
                                                        size=(mata.size(0), chunk_col_size))
                # matb_recv = torch.sparse_csr_tensor(matb_recv_crows, matb_recv_cols, matb_recv_values.float(),
                #                                         size=(chunk_col_size, matb.size(1)))
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-csrinst-{name}")


                start_time(start_inner_timer)
                mata_recv = mata_recv.to_sparse_coo()
                # matb_recv = matb_recv.to_sparse_coo()
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-csr2coo-{name}")

                # matc += torch.mm(mata_recv, matb_recv)
                start_time(start_inner_timer)
                matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_recv._indices(), \
                                                            mata_recv._values(), matb_recv_indices, \
                                                            matb_recv_values, mata.size(0), \
                                                            chunk_col_size, matb.size(1), coalesced=True)
                matc_chunk = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, 
                                                        torch.Size([matc.size(0), matc.size(1)]))
                # matc_chunk = matc_chunk.coalesce()
                matc += matc_chunk
                stop_time_add(start_inner_timer, stop_inner_timer, timing_dict, f"spgemm-loc-spspmm-{name}")
                # matc = matc.to_sparse_csr()
            elif mata.layout == torch.sparse_coo:
                matc_chunk_indices, matc_chunk_values = torch_sparse.spspmm(mata_chunk_indices, \
                                                            mata_chunk_values, matb_recv_indices, \
                                                            matb_recv_values, mata.size(0), \
                                                            chunk_col_size, matb.size(1), coalesced=True)
                matc_chunk = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, 
                                                        torch.Size([matc.size(0), matc.size(1)]))
                # matc_chunk = matc_chunk.coalesce()
                matc += matc_chunk
                if f"spgemm-loc-csrinst-{name}" not in timing_dict:
                    timing_dict[f"spgemm-loc-csrinst-{name}"] = []

                if f"spgemm-loc-csr2coo-{name}" not in timing_dict:
                    timing_dict[f"spgemm-loc-csr2coo-{name}"] = []

                if f"spgemm-loc-spspmm-{name}" not in timing_dict:
                    timing_dict[f"spgemm-loc-spspmm-{name}"] = []
        else:
            if f"spgemm-loc-csrinst-{name}" not in timing_dict:
                timing_dict[f"spgemm-loc-csrinst-{name}"] = []

            if f"spgemm-loc-csr2coo-{name}" not in timing_dict:
                timing_dict[f"spgemm-loc-csr2coo-{name}"] = []

            if f"spgemm-loc-spspmm-{name}" not in timing_dict:
                timing_dict[f"spgemm-loc-spspmm-{name}"] = []

        stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-local-spgemm-{name}", barrier=True)
        torch.cuda.nvtx.range_pop() # local-spgemm

    # print(f"send-rowdata: {timing_dict['spgemm-send-rowdata-prob']}")
    # print(f"send-rownnz: {timing_dict['spgemm-send-rownnz-prob']}")
    # print(f"nnzcount: {timing_dict['spgemm-nnzcount-prob']}")

    rank_row_start = rank_c * replication
    rank_row_stop = (rank_c + 1) * replication - 1

    start_time(start_timer)
    matc = matc.to_sparse_csr()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-coo2csr{name}", barrier=True)

    start_time(start_timer)
    matc = csr_allreduce(matc, rank_row_start, rank_row_stop, rank)
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-{name}", barrier=True)

    start_time(start_timer)
    matc = matc.to_sparse_coo()
    stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-csr2coo{name}", barrier=True)
    return matc._indices(), matc._values()

    # # Old sparse reduction
    # torch.cuda.nvtx.range_push("nvtx-matc-coalesce")
    # start_time(start_timer)
    # if matc.layout == torch.sparse_csr:
    #     matc = matc.to_sparse_coo()
    # matc = matc.coalesce()
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-matc-coalesce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # matc-coalesce

    # # Implement sparse allreduce w/ all_gather and padding
    # torch.cuda.nvtx.range_push("nvtx-reduce-nnz")
    # start_time(start_timer)
    # matc_nnz = torch.cuda.IntTensor(1).fill_(matc._nnz())
    # dist.all_reduce(matc_nnz, dist.ReduceOp.MAX, row_groups[rank_c])
    # matc_nnz = matc_nnz.item()
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-nnz-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # reduce-nnz

    # torch.cuda.nvtx.range_push("nvtx-padding")
    # start_time(start_timer)
    # matc_recv_indices = []
    # matc_recv_values = []
    # for i in range(replication):
    #     matc_recv_indices.append(torch.cuda.LongTensor(2, matc_nnz).fill_(0))
    #     matc_recv_values.append(torch.cuda.DoubleTensor(matc_nnz).fill_(0.0))

    # matc_send_indices = torch.cat((torch.cuda.LongTensor(2, matc_nnz - matc._nnz()).fill_(0), matc._indices()), 1)
    # matc_send_values = torch.cat((torch.cuda.DoubleTensor(matc_nnz - matc._nnz()).fill_(0.0), matc._values()))
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-padding-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # padding

    # torch.cuda.nvtx.range_push("nvtx-allgather")
    # start_time(start_timer)
    # dist.all_gather(matc_recv_indices, matc_send_indices, row_groups[rank_c])
    # dist.all_gather(matc_recv_values, matc_send_values, row_groups[rank_c])
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-allgather-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # all-gather

    # torch.cuda.nvtx.range_push("nvtx-preproc-reduce")
    # start_time(start_timer)
    # matc_recv = []
    # for i in range(replication):
    #     # matc_recv.append(torch.sparse_coo_tensor(matc_recv_indices[i], matc_recv_values[i], matc.size()))

    #     # Unclear why this is necessary but it seems to hang otherwise
    #     nnz_mask = matc_recv_values[i] > 0
    #     matc_nnz_indices = matc_recv_indices[i][:, nnz_mask]
    #     matc_nnz_values = matc_recv_values[i][nnz_mask]

    #     matc_recv.append(torch.sparse_coo_tensor(matc_nnz_indices, matc_nnz_values, matc.size()))
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-preproc-reduce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # preproc-reduce

    # torch.cuda.nvtx.range_push("nvtx-reduce")
    # start_time(start_timer)
    # # matc_recv = torch.stack(matc_recv)
    # # matc = torch.sparse.sum(matc_recv, dim=0)

    # gpu = torch.device(f"cuda:{torch.cuda.current_device()}")
    # for i in range(replication):
    #     recv_rank = rank_c * replication + i
    #     if recv_rank != rank:
    #         # matc += matc_recv[i]
    #         matc_indices = matc._indices().int()
    #         matc_recv_indices = matc_recv[i]._indices().int()
    #         matc_values = matc._values().double()

    #         matc_outputs = coogeam_gpu(matc_indices[0,:], matc_indices[1,:], matc_values, 
    #                             matc_recv_indices[0,:], matc_recv_indices[1,:], matc_recv[i]._values(),
    #                             matc.size(0), matc.size(1))
    #         matc_chunk_rows = matc_outputs[0].long()
    #         matc_chunk_cols = matc_outputs[1].long()
    #         matc_chunk_values = matc_outputs[2].double()
    #         matc_chunk_counts = torch.diff(matc_chunk_rows)
    #         matc_chunk_rows = torch.repeat_interleave(
    #                                     torch.arange(0, matc.size(0), device=gpu),
    #                                     matc_chunk_counts)
    #         matc_chunk_indices = torch.stack((matc_chunk_rows, matc_chunk_cols))
    #         matc = sparse_coo_tensor_gpu(matc_chunk_indices, matc_chunk_values, matc.size())
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # reduce

    # torch.cuda.nvtx.range_push("nvtx-reduce-coalesce")
    # start_time(start_timer)
    # matc = matc.coalesce()
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-reduce-coalesce-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # reduce-coalesce

    # torch.cuda.nvtx.range_push("nvtx-unpad")
    # start_time(start_timer)
    # # nnz_mask = matc._values() != -1.0
    # nnz_mask = matc._values() > 0
    # matc_nnz_indices = matc._indices()[:, nnz_mask]
    # matc_nnz_values = matc._values()[nnz_mask]
    # stop_time_add(start_timer, stop_timer, timing_dict, f"spgemm-unpad-{name}", barrier=True)
    # torch.cuda.nvtx.range_pop() # unpad
    # 
    # return matc_nnz_indices, matc_nnz_values

def gen_prob_dist(numerator, adj_matrix, mb_count, node_count_total, replication, 
                    rank, size, row_groups, col_groups, sa_masks, sa_recv_buff, 
                    timing_dict, name, timing_arg):

    global timing
    timing = timing_arg

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    # TODO: assume n_layers=1 for now
    # start_time(start_timer)
    start_timer.record()
    torch.cuda.nvtx.range_push("nvtx-probability-spgemm")
    p_num_indices, p_num_values = dist_saspgemm15D(
                                        numerator, adj_matrix, replication, rank, size, 
                                        row_groups, col_groups, "prob", sa_masks, 
                                        sa_recv_buff, timing_dict, name)
    torch.cuda.nvtx.range_pop()
    stop_timer.record()
    torch.cuda.synchronize()
    time_taken = start_timer.elapsed_time(stop_timer)
    # timing_dict["probability-spgemm"].append(stop_time(start_timer, stop_timer))
    timing_dict["probability-spgemm"].append(time_taken)

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-compute-p")
    p_den = torch.cuda.DoubleTensor(numerator.size(0)).fill_(0)
    if name == "ladies":
        p_num_values = torch.square(p_num_values)
    elif name == "sage":
        p_num_values = torch.cuda.DoubleTensor(p_num_values.size(0)).fill_(1.0)
    scatterd_add_gpu(p_den, p_num_indices[0, :], p_num_values, p_num_values.size(0))
    # p = torch.sparse_coo_tensor(indices=p_num_indices, 
    #                                 values=p_num_values, 
    #                                 size=(numerator.size(0), node_count_total))
    p = sparse_coo_tensor_gpu(p_num_indices, p_num_values, torch.Size([numerator.size(0), node_count_total]))
    # print(f"p: {p}")
    print(f"p.nnz: {p._nnz()}", flush=True)
    normalize_gpu(p._values(), p_den, p._indices()[0, :], p._nnz())
    timing_dict["compute-p"].append(stop_time(start_timer, stop_timer))
    return p

def sample(p, frontier_size, mb_count, node_count_total, n_darts, replication, 
                rank, size, row_groups, col_groups, timing_dict, name):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    sample_start_timer = torch.cuda.Event(enable_timing=True)
    sample_stop_timer = torch.cuda.Event(enable_timing=True)

    select_start_timer = torch.cuda.Event(enable_timing=True)
    select_stop_timer = torch.cuda.Event(enable_timing=True)

    select_iter_start_timer = torch.cuda.Event(enable_timing=True)
    select_iter_stop_timer = torch.cuda.Event(enable_timing=True)

    rank_c = rank // replication
    rank_col = rank % replication

    n_darts_col = n_darts // replication
    if rank_col == replication - 1:
        n_darts_col = n_darts - (replication - 1) * n_darts_col
    n_darts_col = n_darts

    next_frontier = torch.sparse_coo_tensor(indices=p._indices(),
                                        values=torch.cuda.LongTensor(p._nnz()).fill_(0),
                                        size=(p.size(0), node_count_total))
    # next_frontier = sparse_coo_tensor_gpu(p._indices(), torch.cuda.LongTensor(p._nnz()).fill_(0), 
    #                                             torch.Size([p.size(0), node_count_total]))

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-pre-loop")

    frontier_nnz_sizes = torch.cuda.IntTensor(p.size(0)).fill_(0)
    ones = torch.cuda.IntTensor(next_frontier._indices()[0,:].size(0)).fill_(1)
    frontier_nnz_sizes.scatter_add_(0, next_frontier._indices()[0,:], ones)
    zero_count = (next_frontier._indices()[0,:] == 0).nonzero().size(0)
    frontier_nnz_sizes[0] = zero_count

    torch.cuda.nvtx.range_pop()
    timing_dict["pre-loop"].append(stop_time(start_timer, stop_timer))
    sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)

    iter_count = 0
    selection_iter_count = 0
    torch.cuda.nvtx.range_push("nvtx-sampling")
    
    # underfull_minibatches = (sampled_count < frontier_size).any()
    underfull_minibatches = True

    p_rowsum = torch.cuda.DoubleTensor(p.size(0)).fill_(0)
    while underfull_minibatches:
        iter_count += 1
        start_time(sample_start_timer)
        torch.cuda.nvtx.range_push("nvtx-sampling-iter")

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-prob-rowsum")

        if name == "ladies":
            ps_p_values = torch.cumsum(p._values(), dim=0).roll(1)
            ps_p_values[0] = 0
            p_rowsum.fill_(0)
            p_rowsum.scatter_add_(0, p._indices()[0, :], p._values())
            ps_p_rowsum = torch.cumsum(p_rowsum, dim=0).roll(1)
            ps_p_rowsum[0] = 0
        elif name == "sage":
            ps_p_values = torch.cumsum(p._values(), dim=0)
            p_rowsum.fill_(0)
            p_rowsum.scatter_add_(0, p._indices()[0, :], p._values())
            # ps_p_rowsum = torch.cumsum(p_rowsum, dim=0)
            ps_p_rowsum = torch.cumsum(p_rowsum, dim=0).roll(1)
            ps_p_rowsum[0] = 0
        torch.cuda.nvtx.range_pop()
        timing_dict["prob-rowsum"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-gen-darts")
        # dart_values = torch.cuda.DoubleTensor(n_darts * mb_count).uniform_()
        dart_values = torch.cuda.DoubleTensor(n_darts_col * p.size(0)).uniform_()
        torch.cuda.nvtx.range_pop()
        timing_dict["gen-darts"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-dart-throw")
        # compute_darts1d_gpu(dart_values, n_darts, mb_count)
        # compute_darts1d_gpu(dart_values, n_darts_col, mb_count)
        compute_darts1d_gpu(dart_values, p_rowsum, ps_p_rowsum, n_darts_col, p.size(0))

        torch.cuda.nvtx.range_pop()
        timing_dict["dart-throw"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-filter-darts")
        dart_hits_count = torch.cuda.IntTensor(p._nnz()).fill_(0)
        throw_darts1d_gpu(dart_values, ps_p_values, dart_hits_count, \
                                n_darts_col * p.size(0), p._nnz())
                                # n_darts * mb_count, p._nnz())
        torch.cuda.nvtx.range_pop()
        timing_dict["filter-darts"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-add-to-frontier")
        dist.all_reduce(dart_hits_count, group=row_groups[rank_c])
        next_frontier_values = torch.logical_or(
                                    dart_hits_count, 
                                    next_frontier._values().int()).int()
        # next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
        #                                             values=next_frontier_values,
        #                                             size=(p.size(0), node_count_total))
        next_frontier_tmp = sparse_coo_tensor_gpu(next_frontier._indices(), next_frontier_values,
                                                    torch.Size([p.size(0), node_count_total]))
        torch.cuda.nvtx.range_pop()
        timing_dict["add-to-frontier"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-count-samples")
        # sampled_count = torch.sparse.sum(next_frontier_tmp, dim=1)._values()
        sampled_count = sampled_count.fill_(0)
        sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)
        next_frontier_nnzmask = next_frontier_values.nonzero().squeeze()
        next_frontier_nnzvals = next_frontier_values[next_frontier_nnzmask]
        next_frontier_nnzidxs = next_frontier_tmp._indices()[0, next_frontier_nnzmask]
        # sampled_count.scatter_add_(0, next_frontier_tmp._indices()[0, :], next_frontier_values)
        sampled_count.scatter_add_(0, next_frontier_nnzidxs, next_frontier_nnzvals)
        torch.cuda.nvtx.range_pop()
        timing_dict["count-samples"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-select-preproc")
        overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
        overflowed_minibatches = (overflow > 0).any()
        timing_dict["select-preproc"].append(stop_time(start_timer, stop_timer))

        start_time(select_start_timer)
        torch.cuda.nvtx.range_push("nvtx-dart-selection")

        if rank_col == 0 and overflowed_minibatches:
            while overflowed_minibatches:
                start_time(select_iter_start_timer)
                torch.cuda.nvtx.range_push("nvtx-selection-iter")

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-psoverflow")
                selection_iter_count += 1
                ps_overflow = torch.cumsum(overflow, dim=0)
                total_overflow = ps_overflow[-1].item()
                # n_darts_select = total_overflow // replication
                # if rank_col == replication - 1:
                #     n_darts_select = total_overflow - (replication - 1) * n_darts_select
                n_darts_select = total_overflow
                timing_dict["select-psoverflow"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-reciprocal")
                dart_hits_inv = dart_hits_count.reciprocal().double()
                dart_hits_inv[dart_hits_inv == float("inf")] = 0
                timing_dict["select-reciprocal"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-instmtx")
                # dart_hits_inv_mtx = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                #                                                 values=dart_hits_inv,
                #                                                 size=(p.size(0), node_count_total))
                dart_hits_inv_mtx = sparse_coo_tensor_gpu(next_frontier._indices(), dart_hits_inv,
                                                                torch.Size([p.size(0), node_count_total]))
                timing_dict["select-instmtx"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-invsum")
                dart_hits_inv_sum = torch.cuda.DoubleTensor(p.size(0)).fill_(0)
                # dart_hits_inv_sum.scatter_add_(0, next_frontier._indices()[0, :], dart_hits_inv)
                dart_hits_inv_nnzidxs = dart_hits_inv.nonzero().squeeze()
                indices_nnz_idxs = next_frontier._indices()[0,dart_hits_inv_nnzidxs]
                dart_hits_inv_sum.scatter_add_(0, indices_nnz_idxs, dart_hits_inv[dart_hits_inv_nnzidxs])
                timing_dict["select-invsum"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-psinv")
                ps_dart_hits_inv_sum = torch.cumsum(dart_hits_inv_sum, dim=0).roll(1)
                ps_dart_hits_inv_sum[0] = 0
                timing_dict["select-psinv"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-computedarts")
                # dart_select = torch.cuda.DoubleTensor(total_overflow).uniform_()
                dart_select = torch.cuda.DoubleTensor(n_darts_select).uniform_()
                # Compute darts for selection 
                compute_darts_select_gpu(dart_select, dart_hits_inv_sum, ps_dart_hits_inv_sum, ps_overflow,
                                                p.size(0), n_darts_select)
                                                # mb_count, total_overflow)
                timing_dict["select-computedarts"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-throwdarts")
                # Throw selection darts 
                ps_dart_hits_inv = torch.cumsum(dart_hits_inv, dim=0)
                # throw_darts_select_gpu(dart_select, ps_dart_hits_inv, dart_hits_count, total_overflow,
                throw_darts_select_gpu(dart_select, ps_dart_hits_inv, dart_hits_count, n_darts_select,
                                            next_frontier._nnz())
                timing_dict["select-throwdarts"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-add-to-frontier")
                # dist.all_reduce(dart_hits_count, op=dist.ReduceOp.MIN, group=row_groups[rank_c])
                next_frontier_values = torch.logical_or(dart_hits_count, next_frontier._values()).int()
                # next_frontier_tmp = torch.sparse_coo_tensor(indices=next_frontier._indices(),
                #                                                 values=next_frontier_values,
                #                                                 size=(p.size(0), node_count_total))
                next_frontier_tmp = sparse_coo_tensor_gpu(next_frontier._indices(), next_frontier_values,
                                                                torch.Size([p.size(0), node_count_total]))
                timing_dict["select-add-to-frontier"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-samplecount")
                sampled_count = torch.cuda.IntTensor(p.size(0)).fill_(0)
                sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)
                # sampled_count.scatter_add_(0, next_frontier_tmp._indices()[0,:], next_frontier_tmp._values())
                next_frontier_nnzvals = next_frontier_tmp._values().nonzero().squeeze()
                next_frontier_nnzidxs = next_frontier_tmp._indices()[0,next_frontier_nnzvals]
                sampled_count.scatter_add_(0, next_frontier_nnzidxs, \
                                                next_frontier_tmp._values()[next_frontier_nnzvals])
                timing_dict["select-samplecount"].append(stop_time(start_timer, stop_timer))

                start_time(start_timer)
                torch.cuda.nvtx.range_push("nvtx-select-overflow")
                overflow = torch.clamp(sampled_count - frontier_size, min=0).int()
                overflowed_minibatches = (overflow > 0).any()
                timing_dict["select-overflow"].append(stop_time(start_timer, stop_timer))

                timing_dict["select-iter"].append(stop_time(select_iter_start_timer, select_iter_stop_timer))
        else:
            timing_dict["select-psoverflow"] = []
            timing_dict["select-reciprocal"] = []
            timing_dict["select-instmtx"] = []
            timing_dict["select-invsum"] = []
            timing_dict["select-psinv"] = []
            timing_dict["select-computedarts"] = []
            timing_dict["select-throwdarts"] = []
            timing_dict["select-add-to-frontier"] = []
            timing_dict["select-samplecount"] = []
            timing_dict["select-overflow"] = []
            timing_dict["select-iter"] = []

        dist.barrier(group=row_groups[rank_c])

        torch.cuda.nvtx.range_pop() # nvtx-dart-selection
        timing_dict["dart-selection"].append(stop_time(select_start_timer, select_stop_timer))

        # dist.all_reduce(dart_hits_count, group=row_groups[rank_c])
        dist.broadcast(dart_hits_count, src=rank_c * replication, group=row_groups[rank_c])
        next_frontier_values = torch.logical_or(
                                        dart_hits_count, 
                                        next_frontier._values()).int()
        # next_frontier = torch.sparse_coo_tensor(indices=next_frontier._indices(),
        #                                         values=next_frontier_values,
        #                                         size=(p.size(0), node_count_total))
        next_frontier = sparse_coo_tensor_gpu(next_frontier._indices(), next_frontier_values,
                                                torch.Size([p.size(0), node_count_total]))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-set-probs")
        dart_hits_mask = dart_hits_count > 0
        p._values()[dart_hits_mask] = 0.0

        filled_minibatches = sampled_count == frontier_size
        filled_minibatches_mask = torch.gather(filled_minibatches, 0, p._indices()[0,:])
        p._values()[filled_minibatches_mask] = 0
        torch.cuda.nvtx.range_pop()
        timing_dict["set-probs"].append(stop_time(start_timer, stop_timer))

        start_time(start_timer)
        torch.cuda.nvtx.range_push("nvtx-compute-bool")
        next_frontier_nnzvals = next_frontier._values().nonzero().squeeze()
        next_frontier_nnzidxs = next_frontier._indices()[0,next_frontier_nnzvals]
        sampled_count = torch.cuda.IntTensor(p.size(0)).fill_(0)
        sampled_count = torch.clamp(frontier_size - frontier_nnz_sizes, min=0)
        sampled_count.scatter_add_(0, next_frontier_nnzidxs, \
                                    next_frontier._values()[next_frontier_nnzvals].int())
        underfull_minibatches = (sampled_count < frontier_size).any()
        torch.cuda.nvtx.range_pop()
        timing_dict["compute-bool"].append(stop_time(start_timer, stop_timer))

        overflow = torch.clamp(sampled_count - frontier_size, min=0).int()

        torch.cuda.nvtx.range_pop() # nvtx-sampling-iter
        timing_dict["sampling-iters"].append(stop_time(sample_start_timer, sample_stop_timer))

    torch.cuda.nvtx.range_pop() # nvtx-sampling
    print(f"iter_count: {iter_count}")
    print(f"selection_iter_count: {selection_iter_count}")

    return next_frontier

def select(next_frontier, adj_matrix, batches, sa_masks, sa_recv_buff, nnz, \
                batch_size, frontier_size, mb_count, mb_count_total, node_count_total, replication, \
                rank, size, row_groups, col_groups, timing_dict, layer_id, name):

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-construct-nextf")
    if name == "ladies":
        next_frontier_select = torch.masked_select(next_frontier._indices()[1,:], \
                                                next_frontier._values().bool()).view(mb_count, frontier_size)
        batches_select = torch.masked_select(batches._indices()[1,:], \
                                                batches._values().bool()).view(mb_count, batch_size)
    elif name == "sage":
        next_frontier_select = next_frontier._indices()[1,:].view(mb_count * batch_size, frontier_size)
        batches_select = torch.masked_select(batches._indices()[1,:], \
                                                batches._values().bool()).view(mb_count * batch_size, 1)
    next_frontier_select = torch.cat((next_frontier_select, batches_select), dim=1)
    torch.cuda.nvtx.range_pop()
    timing_dict["construct-nextf"].append(stop_time(start_timer, stop_timer))

    torch.cuda.nvtx.range_push("nvtx-select-rowcols")
    # 1. Make the row/col select matrices
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-select-mtxs")

    if layer_id == 0:
        row_select_mtx_indices = torch.stack((torch.arange(start=0, end=nnz * mb_count).cuda(),
                                              batches_select.view(-1)))
    else:
        # TODO i > 0 (make sure size args to spspmm are right)
        print("ERROR i > 0")

    row_select_mtx_values = torch.cuda.DoubleTensor(row_select_mtx_indices[0].size(0)).fill_(1.0)

    repeated_next_frontier = next_frontier_select.clone()
    if name == "ladies":
        scale_mtx = torch.arange(start=0, end=node_count_total * mb_count, step=node_count_total).cuda()
    elif name == "sage":
        scale_mtx = torch.arange(start=0, end=node_count_total * mb_count * batch_size, 
                                        step=node_count_total).cuda()
    scale_mtx = scale_mtx[:, None]
    repeated_next_frontier.add_(scale_mtx)
    col_select_mtx_rows = repeated_next_frontier.view(-1)
    if name == "ladies":
        col_select_mtx_cols = torch.arange(next_frontier_select.size(1)).cuda().repeat(
                                        mb_count,1).view(-1)
    elif name == "sage":
        col_select_mtx_cols = torch.arange(next_frontier_select.size(1) * batch_size).cuda().repeat(
                                        mb_count,1).view(-1)
    col_select_mtx_indices = torch.stack((col_select_mtx_rows, col_select_mtx_cols))

    # col_select_mtx_values = torch.cuda.DoubleTensor(col_select_mtx_rows.size(0)).fill_(1.0)

    col_select_mtx_values = torch.cuda.DoubleTensor(col_select_mtx_rows.size(0)).fill_(0.0)
    col_unique_rows, col_inverse = torch.unique(col_select_mtx_rows, sorted=True, return_inverse=True)
    col_rows_perm = torch.arange(col_inverse.size(0), dtype=col_inverse.dtype, device=col_inverse.device)
    col_row_mask = col_inverse.new_empty(col_unique_rows.size(0)).scatter_(0, col_inverse, col_rows_perm)
    col_select_mtx_values[col_row_mask] = 1.0

    torch.cuda.nvtx.range_pop()
    timing_dict["select-mtxs"].append(stop_time(start_timer, stop_timer))

    # 2. Multiply row_select matrix with adj_matrix
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-row-select-spgemm")
    # row_select_mtx = torch.sparse_coo_tensor(row_select_mtx_indices, row_select_mtx_values, 
    #                                                 size=(nnz * mb_count, node_count_total))
    row_select_mtx = sparse_coo_tensor_gpu(row_select_mtx_indices, row_select_mtx_values, 
                                                    torch.Size([nnz * mb_count, node_count_total]))
    sa_masks.fill_(0)
    sa_recv_buff.fill_(0)
    sampled_indices, sampled_values = dist_saspgemm15D(row_select_mtx, adj_matrix, replication, rank, size, \
                                                        row_groups, col_groups, "rowsel", sa_masks, 
                                                        sa_recv_buff, timing_dict)
    # sample_mtx = torch.sparse_coo_tensor(sampled_indices, sampled_values, 
    #                                         size=(nnz * mb_count, node_count_total))
    sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                                            torch.Size([nnz * mb_count, node_count_total]))

    torch.cuda.nvtx.range_pop()
    timing_dict["row-select-spgemm"].append(stop_time(start_timer, stop_timer))

    # 3. Expand sampled rows
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-row-select-expand")
    row_shift = torch.cuda.LongTensor(sampled_values.size(0)).fill_(0)
    if name == "ladies":
        shift_rowselect_gpu(row_shift, sampled_indices[0,:], sampled_values.size(0), 
                                rank, size, replication, batch_size, node_count_total, mb_count_total, 
                                batch_size)
    elif name == "sage":
        shift_rowselect_gpu(row_shift, sampled_indices[0,:], sampled_values.size(0), 
                                rank, size, replication, batch_size, node_count_total, mb_count_total, 1)
    sampled_indices[1,:] += row_shift
    torch.cuda.nvtx.range_pop()
    timing_dict["row-select-expand"].append(stop_time(start_timer, stop_timer))

    # 4. Multiply sampled rows with col_select matrix
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-col-select-spgemm")
    if name == "ladies":
        # sample_mtx = torch.sparse_coo_tensor(sampled_indices, sampled_values, 
        #                                         size=(nnz * mb_count, node_count_total * mb_count_total))
        sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                                            torch.Size([nnz * mb_count, node_count_total * mb_count_total]))
    elif name == "sage":
        # sample_mtx = torch.sparse_coo_tensor(sampled_indices, sampled_values, 
        #                                         size=(nnz * mb_count, node_count_total * mb_count_total * nnz))
        sample_mtx = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                                        torch.Size([nnz * mb_count, node_count_total * mb_count_total * nnz]))
    if name == "ladies":
        # col_select_mtx = torch.sparse_coo_tensor(col_select_mtx_indices, col_select_mtx_values,
        #                                         size=(node_count_total * mb_count, next_frontier.size(1)))
        col_select_mtx = sparse_coo_tensor_gpu(col_select_mtx_indices, col_select_mtx_values,
                                                torch.Size([node_count_total * mb_count, next_frontier.size(1)]))
    elif name == "sage":
        # col_select_mtx = torch.sparse_coo_tensor(col_select_mtx_indices, col_select_mtx_values,
        #                     size=(node_count_total * mb_count * batch_size, next_frontier.size(1) * batch_size))
        col_select_mtx = torch.sparse_coo_tensor(col_select_mtx_indices, col_select_mtx_values,
                torch.Size([node_count_total * mb_count * batch_size, next_frontier_select.size(1) * batch_size]))
    sampled_indices, sampled_values = dist_spgemm15D(sample_mtx, col_select_mtx, replication, rank, size, \
                                                        row_groups, col_groups, "colsel", timing_dict)
    torch.cuda.nvtx.range_pop()
    timing_dict["col-select-spgemm"].append(stop_time(start_timer, stop_timer))

    # 5. Store sampled matrices
    start_time(start_timer)
    torch.cuda.nvtx.range_push("nvtx-set-sample")
    # adj_matrix_sample = torch.sparse_coo_tensor(indices=sampled_indices, values=sampled_values, \
    #                                             size=(nnz * mb_count, next_frontier.size(1)))
    adj_matrix_sample = sparse_coo_tensor_gpu(sampled_indices, sampled_values, 
                                            torch.Size([nnz * mb_count, next_frontier.size(1)]))

    torch.cuda.nvtx.range_pop()
    timing_dict["set-sample"].append(stop_time(start_timer, stop_timer))

    torch.cuda.nvtx.range_pop() # nvtx-select-mtxs

    return batches_select, next_frontier_select, adj_matrix_sample
