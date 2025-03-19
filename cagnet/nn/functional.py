from cagnet.partitionings import Partitioning
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

def proc_row_size(size, partitioning):
    if partitioning == Partitioning.TWOD:
        return math.floor(math.sqrt(size))
    elif partitioning == Partitioning.THREED:
        cube_root = int(size ** (1./ 3.))
        if cube_root ** 3 == size:
            return cube_root
        elif (cube_root + 1) ** 3 == size:
            return cube_root + 1
        else:
            print(f"CUBE ROOT ERROR")

def proc_col_size(size, partitioning):
    if partitioning == Partitioning.TWOD:
        return math.floor(math.sqrt(size))
    elif partitioning == Partitioning.THREED:
        cube_root = int(size ** (1./ 3.))
        if cube_root ** 3 == size:
            return cube_root
        elif (cube_root + 1) ** 3 == size:
            return cube_root + 1
        else:
            print(f"CUBE ROOT ERROR")

def proc_c_size(size, partitioning):
    if partitioning == Partitioning.THREED:
        cube_root = int(size ** (1./ 3.))
        if cube_root ** 3 == size:
            return cube_root
        elif (cube_root + 1) ** 3 == size:
            return cube_root + 1
        else:
            print(f"CUBE ROOT ERROR")

def relu(x, partitioning):
    return F.relu(x)

def log_softmax(self, x, partitioning, dim=1):
    if partitioning == Partitioning.ONED or partitioning == Partitioning.ONE5D:
        return F.log_softmax(x, dim)
    elif partitioning == Partitioning.TWOD:
        return LogSoftmaxTWOD.apply(self, x)

def nll_loss(logits_rank, labels_rank, node_count, total_classes, partitioning, rank, group, size, \
                            train_mask=None, partitions=None):
    # print(f"logits_rank: {logits_rank}")
    # print(f"labels_rank: {labels_rank}")
    if partitioning == Partitioning.ONED or partitioning == Partitioning.ONE5D:
        loss = F.nll_loss(logits_rank, labels_rank, reduction="sum") 
        # print(loss)
        loss_recv = []
        for i in range(size):
            loss_recv.append(torch.cuda.FloatTensor(1))
        dist.all_gather(loss_recv, loss, group)
        loss_recv[rank] = loss
        loss = sum(loss_recv) / node_count
        return loss

def cross_entropy(logits_rank, labels_rank, node_count, total_classes, partitioning, rank, group, size, \
                            train_mask=None, partitions=None):
    # print(f"logits_rank: {logits_rank}")
    # print(f"labels_rank: {labels_rank}")
    if partitioning == Partitioning.ONED or partitioning == Partitioning.ONE5D:
        loss = F.cross_entropy(logits_rank, labels_rank, reduction="sum") 
        # print(loss)
        loss_recv = []
        for i in range(size):
            loss_recv.append(torch.cuda.FloatTensor(1))
        dist.all_gather(loss_recv, loss, group)
        loss_recv[rank] = loss
        loss = sum(loss_recv) / node_count
        return loss

    elif partitioning == Partitioning.TWOD:
        proc_row = proc_row_size(size, partitioning)
        proc_col = proc_col_size(size, partitioning)

        rank_row = int(rank / proc_col)
        rank_col = rank % proc_col

        class_per_rank = total_classes // proc_col

        row_groups = group[0]
        col_groups = group[1]

        min_class = rank_col * class_per_rank
        max_class = min((rank_col + 1) * class_per_rank, total_classes)
        if rank_col == proc_col - 1:
            max_classes = total_classes

        if list(labels_rank.size())[0] > 0:
            datay_ids = labels_rank.long()

            filtered_indices = torch.mul(datay_ids >= min_class, datay_ids < max_class).float()
            indices = torch.nonzero(filtered_indices * \
                                torch.cuda.FloatTensor(datay_ids.size()).fill_(1)).squeeze()

            datay_ids = labels_rank.long().view(-1, 1)
            datay_ids = datay_ids.index_select(0, indices)
            datay_ids -= min_class
            outputs_ids = logits_rank.index_select(0, indices)

            classes = torch.gather(outputs_ids, 1, datay_ids)
            loss_calc = torch.sum(classes)
            loss_calc_tens = torch.Tensor([loss_calc.item()])

            rank_row_src = rank_row * proc_col

            dist.reduce(loss_calc, dst=rank_row_src, op=dist.reduce_op.SUM, group=row_groups[rank_row])
            dist.broadcast(loss_calc, src=rank_row_src, group=row_groups[rank_row]) 

            vertex_train_count = (train_mask.size(0) - (train_mask == 0).sum(dim=0))
            loss_calc = -loss_calc / vertex_train_count
            return loss_calc

        else:
            fake_loss = (logits_rank * torch.cuda.FloatTensor(logits_rank.size()).fill_(0)).sum()
            return fake_loss

    elif partitioning == Partitioning.THREED:
        proc_row = proc_row_size(size, partitioning)
        proc_col = proc_col_size(size, partitioning)
        proc_c = proc_c_size(size, partitioning)

        rank_row = int((rank // proc_c) // proc_col) # i in process grid
        rank_col = int((rank // proc_c) % proc_col)  # j in process grid
        rank_c = rank - (rank_row * (proc_col * proc_c) + rank_col * proc_c) # k in process grid

        class_per_rank = total_classes // proc_col

        row_groups = group[0]
        col_groups = group[1]
        c_groups = group[2]

        min_class = rank_col * class_per_rank
        max_class = min((rank_col + 1) * class_per_rank, total_classes)
        if rank_col == proc_col - 1:
            max_class = total_classes

        # Note: bool type removes warnings, unsure of perf penalty
        if list(labels_rank.size())[0] > 0:
            datay_ids = labels_rank.long()

            filtered_indices = torch.mul(datay_ids >= min_class, datay_ids < max_class).float()
            indices = torch.nonzero(filtered_indices * \
                            torch.cuda.FloatTensor(datay_ids.size()).fill_(1)).squeeze()

            datay_ids = labels_rank.long().view(-1, 1)
            datay_ids = datay_ids.index_select(0, indices)
            datay_ids -= min_class
            outputs_ids = logits_rank.index_select(0, indices)

            classes = torch.gather(outputs_ids, 1, datay_ids)
            loss_calc = torch.sum(classes)
            loss_calc_tens = torch.Tensor([loss_calc.item()])

            dist.all_reduce(loss_calc, op=dist.reduce_op.SUM, group=row_groups[rank_row][rank_c])

            vertex_train_count = (train_mask.size(0) - (train_mask == 0).sum(dim=0))
            loss_calc = -loss_calc / vertex_train_count

            return loss_calc
        else:
            fake_loss = (logits_rank * torch.cuda.FloatTensor(logits_rank.size()).fill_(0)).sum()
            return fake_loss


class LogSoftmaxTWOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, z):
        maxes = torch.max(z, dim=1, keepdim=True)[0]
        maxes_recv = []
        for i in range(self.proc_col):
            maxes_recv.append(torch.cuda.FloatTensor(maxes.size(), device=self.device))

        dist.all_gather(maxes_recv, maxes, group=self.row_groups[self.rank_row])
        maxes_recv[self.rank_col] = maxes
        maxes = torch.max(torch.cat(maxes_recv, dim=1), dim=1, keepdim=True)[0]

        h = torch.exp(z - maxes)
        sm_sum = torch.sum(h, dim=1, keepdim=True)

        sm_sum_recv = []
        for i in range(self.proc_col):
            sm_sum_recv.append(torch.cuda.FloatTensor(sm_sum.size(), device=self.device))

        dist.all_gather(sm_sum_recv, sm_sum, group=self.row_groups[self.rank_row])
        sm_sum_recv[self.rank_col] = sm_sum
        sm_sum = torch.sum(torch.cat(sm_sum_recv, dim=1), dim=1, keepdim=True)
        sm_sum = torch.log(sm_sum)
        h = z - maxes - sm_sum

        ctx.self = self
        ctx.z = z
        return h

    @staticmethod
    def backward(ctx, grad_output):
        self = ctx.self
        z = ctx.z

        width = self.n_classes

        with torch.set_grad_enabled(True):
            chunk_sizes_col = []
            width_per_col = width // self.proc_col

            for i in range(self.proc_col):
                if i == self.proc_col - 1:
                    chunk_sizes_col.append(width - width_per_col * (self.proc_col - 1))
                else:
                    chunk_sizes_col.append(width_per_col)

            width_per_proc = width - width_per_col * (self.proc_col - 1)
            if z.size(1) != width_per_proc:
                z = torch.cat((z, torch.cuda.FloatTensor(z.size(0), width_per_proc - z.size(1))), dim=1)

            z_recv = []
            for i in range(self.proc_col):
                z_recv.append(torch.cuda.FloatTensor(z.size()))

            dist.all_gather(z_recv, z, group=self.row_groups[self.rank_row])
            z_recv[self.rank_col] = z

            for i in range(self.proc_col - 1):
                pad_col = width // self.proc_col
                z_recv[i] = z_recv[i][:,:pad_col]

            z = torch.cat(z_recv, dim=1)

            if grad_output is not None:
                if grad_output.size(1) != width_per_proc:
                    grad_output = torch.cat((grad_output, 
                                                torch.cuda.FloatTensor(grad_output.size(0), 
                                                                width_per_proc - grad_output.size(1))), 
                                                dim=1)

                grad_output_recv = []
                for i in range(self.proc_col):
                    grad_output_recv.append(torch.cuda.FloatTensor(grad_output.size()))

                dist.all_gather(grad_output_recv, grad_output, group=self.row_groups[self.rank_row])
                grad_output_recv[self.rank_col] = grad_output

                for i in range(self.proc_col - 1):
                    pad_col = width // self.proc_col
                    grad_output_recv[i] = grad_output_recv[i][:,:pad_col]

                grad_output = torch.cat(grad_output_recv, dim=1)

            maxes = torch.max(z, dim=1, keepdim=True)[0]
            h = torch.exp(z - maxes)
            sm_sum = torch.sum(h, dim=1, keepdim=True)
            sm_sum = torch.log(sm_sum)

            h = z - maxes - sm_sum

            func_eval = h
            z_gathered = z
            grad_output_gathered = grad_output

            width = z_gathered.size(1)

            sigmap = torch.autograd.grad(outputs=func_eval, \
                                            inputs=z_gathered, \
                                            grad_outputs=grad_output_gathered)[0]

            chunk_sizes_col = []
            sigmap_per_col = width // self.proc_col

            for i in range(self.proc_col):
                if i == self.proc_col - 1:
                    chunk_sizes_col.append(width - sigmap_per_col * (self.proc_col - 1))
                else:
                    chunk_sizes_col.append(sigmap_per_col)

            z_grad = sigmap.split(chunk_sizes_col, dim=1)[self.rank_col]

        return None, z_grad

