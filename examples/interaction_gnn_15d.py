import argparse
import copy
import math
import os
import os.path as osp
import time
import torch.multiprocessing as mp
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch_geometric
import torch_geometric.transforms as T
import torch_sparse

from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.loader import DataLoader, NeighborSampler, NeighborLoader
from torch_geometric.loader.shadow import ShaDowKHopSampler
# from shadow import ShaDowKHopSampler
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch.multiprocessing import Process
from torchmetrics.classification import Precision, Recall, PrecisionRecallCurve, ROC, AUROC, Accuracy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
from cagnet.samplers import shadow_sampler
from cagnet.samplers.utils import *
import cagnet.nn.functional as CAGF
import torch.nn.functional as F

from datetime import timedelta
from sparse_coo_tensor_cpp import sort_dst_proc_gpu
from sklearn.metrics import roc_auc_score
from statistics import median

import socket
import wandb
import yaml

class InteractionGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, 
                    nb_node_layer, nb_edge_layer, n_graph_iters, 
                    node_features, edge_features, layernorm, batchnorm, 
                    hidden_activation, output_activation, 
                    aggr, checkpointing, replication, 
                    impl="cagnet", dataset="physics_ex3"):

        super(InteractionGNN, self).__init__()
        # self.layers = nn.ModuleList()
        self.nb_node_layer = nb_node_layer
        self.nb_edge_layer = nb_edge_layer
        self.n_graph_iters = n_graph_iters
        self.node_features = node_features
        self.edge_features = edge_features
        self.layernorm = layernorm
        self.batchnorm = batchnorm
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.aggr = aggr
        self.checkpointing = checkpointing
        self.replication = replication
        self.timings = dict()
        self.impl = impl
        self.dataset = dataset

        self.timings["sample"] = []
        self.timings["train"] = []
        self.timings["extract"] = []
        self.timings["fwdbwd"] = []
        self.timings["fwd"] = []
        self.timings["bwd"] = []
        self.timings["h2d"] = []
        self.timings["allreduce"] = []
        self.timings["allreduce-pt1"] = []
        self.timings["allreduce-pt2"] = []
        self.timings["allreduce-pt3"] = []
        self.timings["allreduce-pt4"] = []
        self.timings["shadow-sampling"] = []
        self.timings["shadow-selection"] = []
        self.timings["shadow-extraction"] = []
        self.timings["fwd-prelude"] = []
        self.timings["fwd-encoder"] = []
        self.timings["fwd-graphiters"] = []

        torch.manual_seed(0)
        # aggr_list = ["sum", "mean", "max", "std"]
        # self.aggregation = torch_geometric.nn.aggr.MultiAggregation(aggr_list, mode="cat")

        # network_input_size = (1 + 2 * len(aggr_list)) * n_hidden

        self.node_encoder = self.make_mlp(
            input_size=in_feats,
            sizes=[n_hidden] * nb_node_layer,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
        )
        
        if len(self.edge_features) > 0:
            self.edge_encoder = self.make_mlp(
                input_size=len(self.edge_features),
                sizes=[n_hidden] * nb_edge_layer,
                output_activation=output_activation,
                hidden_activation=hidden_activation,
                layer_norm=layernorm,
                batch_norm=batchnorm,
            )
        else:
            self.edge_encoder = self.make_mlp(
                input_size=2 * n_hidden,
                sizes=[n_hidden] * nb_edge_layer,
                output_activation=output_activation,
                hidden_activation=hidden_activation,
                layer_norm=layernorm,
                batch_norm=batchnorm,
            )

        torch.manual_seed(0)
        # aggr_list = ["sum", "mean", "max", "std"]
        # self.aggregation = torch_geometric.nn.aggr.MultiAggregation(aggr_list, mode="cat")

        # network_input_size = (1 + 2 * len(aggr_list)) * n_hidden

        self.node_encoder = self.make_mlp(
            input_size=in_feats,
            sizes=[n_hidden] * nb_node_layer,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
        )
        
        if len(self.edge_features) > 0:
            self.edge_encoder = self.make_mlp(
                input_size=len(self.edge_features),
                sizes=[n_hidden] * nb_edge_layer,
                output_activation=output_activation,
                hidden_activation=hidden_activation,
                layer_norm=layernorm,
                batch_norm=batchnorm,
            )
        else:
            self.edge_encoder = self.make_mlp(
                input_size=2 * n_hidden,
                sizes=[n_hidden] * nb_edge_layer,
                output_activation=output_activation,
                hidden_activation=hidden_activation,
                layer_norm=layernorm,
                batch_norm=batchnorm,
            )
        
        in_edge_net = n_hidden * 6
        self.edge_network = nn.ModuleList(
            [
                self.make_mlp(
                    input_size=in_edge_net,
                    sizes=[n_hidden] * nb_edge_layer,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    layer_norm=layernorm,
                    batch_norm=batchnorm,
                )
                for i in range(n_graph_iters)
            ]
        )
        
        in_node_net = n_hidden * 4
        self.node_network = nn.ModuleList(
            [
                self.make_mlp(
                    input_size=in_node_net,
                    sizes=[n_hidden] * nb_node_layer,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    layer_norm=layernorm,
                    batch_norm=batchnorm,
                )
                for i in range(n_graph_iters)
            ]
        )
        
        # edge decoder
        self.edge_decoder = self.make_mlp(
            input_size=n_hidden,
            sizes=[n_hidden] * nb_edge_layer,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
        )

        # edge output transform layer
        self.edge_output_transform = self.make_mlp(
            input_size=n_hidden,
            sizes=[n_hidden, 1],
            output_activation=None,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
        )

        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        
    def make_mlp(
        self, 
        input_size,
        sizes,
        hidden_activation="ReLU",
        output_activation=None,
        layer_norm=False,  # TODO : change name to hidden_layer_norm while ensuring backward compatibility
        output_layer_norm=False,
        batch_norm=False,  # TODO : change name to hidden_batch_norm while ensuring backward compatibility
        output_batch_norm=False,
        input_dropout=0,
        hidden_dropout=0,
        track_running_stats=False,
    ):
        """Construct an MLP with specified fully-connected layers."""
        hidden_activation = getattr(nn, hidden_activation)
        if output_activation is not None:
            output_activation = getattr(nn, output_activation)
        layers = []
        n_layers = len(sizes)
        sizes = [input_size] + sizes
        # Hidden layers
        for i in range(n_layers - 1):
            if i == 0 and input_dropout > 0:
                layers.append(nn.Dropout(input_dropout))
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if layer_norm:  # hidden_layer_norm
                layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
            if batch_norm:  # hidden_batch_norm
                layers.append(
                    nn.BatchNorm1d(
                        sizes[i + 1],
                        eps=6e-05,
                        track_running_stats=track_running_stats,
                        affine=True,
                    )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
                )
            layers.append(hidden_activation())
            if hidden_dropout > 0:
                layers.append(nn.Dropout(hidden_dropout))
        # Final layer
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if output_activation is not None:
            if output_layer_norm:
                layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
            if output_batch_norm:
                layers.append(
                    nn.BatchNorm1d(
                        sizes[-1],
                        eps=6e-05,
                        track_running_stats=track_running_stats,
                        affine=True,
                    )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
                )
            layers.append(output_activation())
        return nn.Sequential(*layers)

    def message_step(self, x, e, src, dst, i):
        # Compute new node features
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
        e_updated = self.edge_network[i](edge_inputs)
        edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
        edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])

        node_inputs = torch.cat(
            [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
        )  # to check : the order dst src  x ?
        x_updated = self.node_network[i](node_inputs)
        
        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        classifier_output = self.output_edge_classifier(classifier_inputs).squeeze(-1)
        return classifier_output

    def forward(self, batch, epoch=0):
        # x = torch.stack([batch["z"]], dim=-1).float()

        start_timer = torch.cuda.Event(enable_timing=True)
        stop_timer = torch.cuda.Event(enable_timing=True)
        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        x = torch.stack(
            [batch[feature] for feature in self.node_features], dim=-1
        ).float()

        if len(self.edge_features) > 0:
            edge_attr = torch.stack(
                [batch[feature] for feature in self.edge_features], dim=-1
            ).float()
        else:
            edge_attr = None

        if batch.edge_index is not None:
            src, dst = batch.edge_index
        elif batch.adj_t is not None:
            coo = batch.adj_t.coo()
            src, dst = coo[0], coo[1]

        x.requires_grad = True
        if edge_attr is not None:
            edge_attr.requires_grad = True

        if epoch >= 1:
            self.timings["fwd-prelude"].append(stop_time(start_timer, stop_timer, timing_arg=True))

        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        if self.checkpointing:
            x = checkpoint(self.node_encoder, x, use_reentrant=False)
            if edge_attr is not None:
                e = checkpoint(self.edge_encoder, edge_attr, use_reentrant=False)
            else:
                e = checkpoint(self.edge_encoder, torch.cat([x[src], x[dst]], dim=-1), use_reentrant=False)
        else:
            x = self.node_encoder(x)
            if edge_attr is not None:
                e = self.edge_encoder(edge_attr)
            else:
                e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        if epoch >= 1:
            self.timings["fwd-encoder"].append(stop_time(start_timer, stop_timer, timing_arg=True))

        # if concat
        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        input_x = x
        input_e = e
        # outputs = []
        outputs = None
        for i in range(self.n_graph_iters):
            # if concat
            if self.checkpointing:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
                x, e, out = checkpoint(self.message_step, x, e, src, dst, i, use_reentrant=False)
                # outputs.append(out)
                outputs = out
            else:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
                x, e, out = self.message_step(x, e, src, dst, i)
                outputs = out
        if epoch >= 1:
            self.timings["fwd-graphiters"].append(stop_time(start_timer, stop_timer, timing_arg=True))

        # return outputs[-1].squeeze(-1)
        return outputs.squeeze(-1)

    def loss_function(self, output, batch, balance="proportional"):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled 
        with the `weighting` config option.
        """

        assert hasattr(batch, "y"), "The batch does not have a truth label"
        assert hasattr(batch, "weights"), "The batch does not have a weighting label"
        
        assert hasattr(batch, "y"), (
            "The batch does not have a truth label. Please ensure the batch has a `y`"
            " attribute."
        )
        assert hasattr(batch, "weights"), (
            "The batch does not have a weighting label. Please ensure the batch"
            " weighting is handled in preprocessing."
        )

        if balance not in ["equal", "proportional"]:
            warnings.warn(
                f"{balance} is not a proper choice for the loss balance. Use either 'equal' or 'proportional'. Automatically switching to 'proportional' instead."
            )
            balance = "proportional"

        negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0)

        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask],
            torch.zeros_like(output[negative_mask]),
            weight=batch.weights[negative_mask].abs(),
            reduction="sum",
        )

        positive_mask = (batch.y == 1) & (batch.weights > 0)
        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask],
            torch.ones_like(output[positive_mask]),
            weight=batch.weights[positive_mask].abs(),
            reduction="sum",
        )

        if balance == "proportional":
            n = positive_mask.sum() + negative_mask.sum()
            return (
                (positive_loss + negative_loss) / n,
                positive_loss.detach() / n,
                negative_loss.detach() / n,
            )
        else:
            n_pos, n_neg = positive_mask.sum(), negative_mask.sum()
            n = n_pos + n_neg
            return (
                positive_loss / n_pos + negative_loss / n_neg,
                positive_loss.detach() / n,
                negative_loss.detach() / n,
            )

    @torch.no_grad()
    def evaluate(self, test_loader, epoch, curr_lr, epoch_time, wandb_arg, rank):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """
        # val_auc = AUROC(task="binary").to(self.device)
        # val_precision = Precision(task="binary").to(self.device)
        # val_recall = Recall(task="binary").to(self.device)    
        val_loss = []
        eff = []
        tar_pur = []
        tot_pur = []

        print("Evaluating")
        test_batches = []
        for i, batch in enumerate(test_loader):
            # batch = batch.to(self.device)
            if rank == 0:
                print(f"batch {i}")
                # batch = batch.to(self.device)
                batch = batch.cuda()
                output = self(batch)
                loss, pos_loss, neg_loss = self.loss_function(output, batch)

                scores = torch.sigmoid(output)
                batch.scores = scores.detach()
                test_batches.append(batch)

                all_truth = batch.y.bool()
                target_truth = (batch.weights > 0) & all_truth


                preds = torch.sigmoid(output) > 0.5
                
                # Positives
                edge_positive = preds.sum().float()

                # Signal true & signal tp
                target_true = target_truth.sum().float()
                target_true_positive = (target_truth.bool() & preds).sum().float()
                all_true_positive = (all_truth.bool() & preds).sum().float()
                target_auc = roc_auc_score(
                    target_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
                )
                # Eff, pur, auc
                target_eff = target_true_positive / target_true
                target_pur = target_true_positive / edge_positive
                total_pur = all_true_positive / edge_positive
                
                val_losses = self.loss_function(output, batch)
                val_losses = sum(val_losses)
                val_loss.append(val_losses)
                eff.append(target_eff.item())
                tar_pur.append(target_pur.item())
                tot_pur.append(total_pur.item())
                # val_precision.update(preds, target_truth)
                # val_recall.update(preds, target_truth)
                # val_auc.update(output, target_truth)

        if rank == 0:
            avg_loss = sum(val_loss) / len(val_loss)
            target_eff = sum(eff) / len(eff)
            avg_tarpur = sum(tar_pur) / len(tar_pur)
            avg_totpur = sum(tot_pur) / len(tot_pur)
            
        # efficiency = val_precision.compute()
        # purity = val_recall.compute()
        # auc = val_auc.compute()
        print(f"Logging")
        if rank == 0:
            if wandb_arg:
                pass
                # wandb.log({
                #     # "train_loss": loss,
                #     "current_lr": curr_lr,
                #     "eff": target_eff,
                #     "target_pur": avg_tarpur,
                #     "total_pur": avg_totpur,
                #     "auc": target_auc,
                #     "val_loss": avg_loss,
                #     "epoch": epoch,
                #     "time": epoch_time,
                #     # "trainer/global_step": step + epoch*step
                #     # "trainer/global_step": epoch
                # })
            print(f"Target Purity: {avg_tarpur}", flush=True)
            print(f"Total Purity: {avg_totpur}", flush=True)
            print(f"Efficiency: {target_eff}", flush=True)
            print(f"AUC: {target_auc}", flush=True)

        # val_auc.reset(), val_precision.reset(), val_recall.reset()

        full_auc = 0.0
        masked_auc = 0.0
        # if self.dataset == "physics_ex3":
        #     filename = f"{self.impl}_{self.dataset}"
        #     full_auc, masked_auc = graph_roc_curve("testset", test_batches, "Interaction GNN ROC curve", filename)
        #     print(f"full_auc: {full_auc} type(full_auc): {type(full_auc)}", flush=True)
        #     print(f"masked_auc: {masked_auc} type(masked_auc): {type(masked_auc)}", flush=True)
        # elif False and self.dataset == "ctd":
        #     filename = f"{self.impl}_{self.dataset}_edgewise"
        #     plot_config = {"title": "GNN edge-wise Efficiency vs (r,z)",
        #                         "filename": filename }
        #     target_tracks = {"pt": [1000, np.inf],
        #                         "nhits": [3, np.inf],
        #                         "primary": True,
        #                         # pdgId: {}
        #                         "radius": [0., 260.],
        #                         "eta_particle": [-4., 4.],
        #                         "redundant_split_edges": False}

        #     config = {"dataset": self.dataset,
        #                 "score_cut": 0.5,
        #                 "target_tracks": target_tracks,
        #                 "stage_dir": filename }
        #     signal_efficiency = gnn_efficiency_rz(test_batches, plot_config, config)
        #     plot_config["vmin"] = 0.4
        #     # gnn_purity_rz(test_batches, plot_config, config)
        #     gnn_purity_rz(test_loader, plot_config, config)
        #     full_auc = signal_efficiency
        #     masked_auc = 0.0
        #     print(f"signal_efficiency: {signal_efficiency}", flush=True)

        return full_auc, masked_auc

        # # Load data from testset directory
        # graph_constructor = cls(config).to(device)
        # if checkpoint is not None:
        #     print(f"Restoring model from {checkpoint}")
        #     graph_constructor = cls.load_from_checkpoint(checkpoint, hparams=config).to(
        #         device
        #     )
        # graph_constructor.setup(stage="test")

        # plots:
        #   graph_scoring_efficiency: 
        #     title: Interaction GNN Edge-wise Efficiency
        #     pt_units: GeV
        #   graph_roc_curve:
        #     title: Interaction GNN ROC curve
        # all_plots = config["plots"]

        ## TODO: Handle the list of plots properly
        #for plot_function, plot_config in all_plots.items():
        #    if hasattr(eval_utils, plot_function):
        #        getattr(eval_utils, plot_function)(
        #            graph_constructor, plot_config, config
        #        )
        #    else:
        #        print(f"Plot {plot_function} not implemented")

    def ignn_allreduce(self, params, size, epoch):

        start_timer = torch.cuda.Event(enable_timing=True)
        stop_timer = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        grad_tensors = []
        tensor_names = []
        param_list = list(params)
        for name, W in param_list:
            if W.grad is not None:
                # print(f"name: {name} W.grad.size(): {W.grad.size()} ndim: {W.grad.ndim}", flush=True)
                if W.grad.size(0) == 64:
                    if W.grad.ndim == 2:
                        grad_tensors.append(W.grad)
                        tensor_names.append(name)
                    elif W.grad.ndim == 1:
                        grad_tensors.append(W.grad.unsqueeze(-1))
                        tensor_names.append(name)
                    # else:
                    #     print(f"tensor error: {name} W.grad.size: {W.grad.size()}")

        stacked_grad_tensors = torch.cat(grad_tensors, dim=1)
        if epoch >= 1:
            self.timings["allreduce-pt1"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))

        stacked_grad_tensors = stacked_grad_tensors.contiguous()
        torch.cuda.synchronize()
        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        dist.all_reduce(stacked_grad_tensors)
        if epoch >= 1:
            self.timings["allreduce-pt2"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))

        torch.cuda.synchronize()
        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        stacked_grad_tensors /= size
        grad_tensor_idx = 0
        name_tensor_idx = 0
        for name, W in param_list:
            if W.grad is not None:
                if W.grad.size(0) == 64:
                    name_tensor_idx += 1
                    if W.grad.ndim == 2:
                        W.grad = stacked_grad_tensors[:,\
                                    grad_tensor_idx:grad_tensor_idx + W.grad.size(1)]
                        grad_tensor_idx += W.grad.size(1)
                    elif W.grad.ndim == 1:
                        W.grad = stacked_grad_tensors[:,\
                                        grad_tensor_idx:grad_tensor_idx + 1].squeeze()
                        grad_tensor_idx += 1
        if epoch >= 1:
            self.timings["allreduce-pt3"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))

        if epoch >= 1:
            start_time(start_timer, timing_arg=True)
        for _, W in params:
            if W.grad is not None and W.grad.size(0) != 64:
                dist.all_reduce(W.grad)
                W.grad /= size
        if epoch >= 1:
            self.timings["allreduce-pt4"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))

def get_proc_groups(rank, size, replication):
    rank_c = rank // replication
     
    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx, normalization):
    if not normalization:
        return adj_part

    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].float(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))

    return adj_part

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sparse.sum(mx, 1)
    r_inv = torch.float_power(rowsum, -1).flatten()
    # r_inv._values = r_inv._values()[torch.isinf(r_inv._values())] = 0.
    # r_mat_inv = torch.diag(r_inv._values())
    r_inv_values = torch.cuda.DoubleTensor(r_inv.size(0)).fill_(0)
    r_inv_values[r_inv._indices()[0,:]] = r_inv._values()
    # r_inv_values = r_inv._values()
    r_inv_values[torch.isinf(r_inv_values)] = 0
    r_mat_inv = torch.sparse_coo_tensor([np.arange(0, r_inv.size(0)).tolist(),
                                     np.arange(0, r_inv.size(0)).tolist()],
                                     r_inv_values,
                                     size=(r_inv.size(0), r_inv.size(0)))
    # mx = r_mat_inv.mm(mx.float())
    mx_indices, mx_values = torch_sparse.spspmm(r_mat_inv._indices(), r_mat_inv._values(), 
                                                    mx._indices(), mx._values(),
                                                    r_mat_inv.size(0), r_mat_inv.size(1), mx.size(1),
                                                    coalesced=True)
    mx = torch.sparse_coo_tensor(indices=mx_indices, values=mx_values.double(), size=(r_mat_inv.size(0), mx.size(1)))
    return mx

def one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, \
                            normalize, replicate_graph):
    node_count = inputs.size(0)
    # n_per_proc = math.ceil(float(node_count) / size)
    # n_per_proc = math.ceil(float(node_count) / (size / replication))
    n_per_proc = node_count // (size // replication)

    am_partitions = None
    am_pbyp = None

    # inputs = inputs.to(torch.device("cpu"))
    # adj_matrix = adj_matrix.to(torch.device("cpu"))
    # torch.cuda.synchronize()

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        if not replicate_graph:
            am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)
        else:
            am_partitions = None
        # # proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        # # am_pbyp, _ = split_coo(am_partitions[rank_c], node_count, n_per_proc, 0)
        # # print(f"before", flush=True)
        # # for i in range(len(am_pbyp)):
        # #     if i == size // replication - 1:
        # #         last_node_count = vtx_indices[i + 1] - vtx_indices[i]
        # #         am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
        # #                                                 size=(last_node_count, proc_node_count),
        # #                                                 requires_grad=False)

        # #         am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
        # #                                         vtx_indices[rank_c], normalize)
        # #     else:
        # #         am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
        # #                                                 size=(n_per_proc, proc_node_count),
        # #                                                 requires_grad=False)

        # #         am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
        # #                                         vtx_indices[rank_c], normalize)

        if not replicate_graph:
            for i in range(len(am_partitions)):
                proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                        torch.ones(am_partitions[i].size(1)).double(), 
                                                        size=(node_count, proc_node_count), 
                                                        requires_grad=False)
                am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i], \
                                                        normalize)
            adj_matrix_loc = am_partitions[rank_c]
        else:
            adj_matrix_loc = None

        # # input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)
        input_partitions = torch.split(inputs, inputs.size(0) // (size // replication), dim=0)
        if len(input_partitions) > (size // replication):
            input_partitions_fused = [None] * (size // replication)
            input_partitions_fused[:-1] = input_partitions[:-2]
            input_partitions_fused[-1] = torch.cat(input_partitions[-2:], dim=0)
            input_partitions = input_partitions_fused

        inputs_loc = input_partitions[rank_c]

    # print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}")
    return inputs_loc, adj_matrix_loc, am_partitions, input_partitions

def one5d_partition_mb(rank, size, batches, replication, batch_size):
    rank_c = rank // replication
    # batch_partitions = torch.split(batches, int(mb_count // (size / replication)), dim=0)
    batch_partitions = torch.split(batches, int(batch_size // (size / replication)), dim=1)
    return batch_partitions[rank_c]
    # batch_partitions = torch.split(batches, int(mb_count // size), dim=0)
    # return batch_partitions[rank]

def main(local_rank, args, trainset, testset, 
                    g_loc_crows, g_loc_cols, g_loc_vals,
                    g_loc_t_crows, g_loc_t_cols, g_loc_t_vals, node_count, edge_count, 
                    node_features, edge_features, num_features, num_classes):

    torch.set_num_threads(1)
    # load and preprocess dataset
    # Initialize distributed environment with SLURM
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        global_rank = int(os.environ["RANK"]) * args.gpu + local_rank

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        global_size = int(os.environ["WORLD_SIZE"]) * args.gpu

    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"
    
    print(f"device_count: {torch.cuda.device_count()}")
    print(f"hostname: {socket.gethostname()}")
    print(f"local_rank: {local_rank}", flush=True)
    if not dist.is_initialized():
        print(f"global_rank: {global_rank} local_rank: {local_rank} size: {global_size} initializing process group")
        dist.init_process_group(backend=args.dist_backend, rank=global_rank, world_size=global_size, timeout=timedelta(minutes=30))
        dist.barrier()
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}")
    torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{rank % args.gpu}')
    g_loc = torch.sparse_csr_tensor(g_loc_crows, g_loc_cols, g_loc_vals)
    g_loc_t = torch.sparse_csr_tensor(g_loc_t_crows, g_loc_t_cols, g_loc_t_vals)
    g_loc = g_loc.to(device)
    g_loc_t = g_loc_t.to(device)
    # trainset = trainset.to(device)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_inner_timer = torch.cuda.Event(enable_timing=True)
    stop_inner_timer = torch.cuda.Event(enable_timing=True)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)

    torch.manual_seed(0)

    proc_row = size // args.replication
    rank_row = rank // args.replication
    rank_col = rank % args.replication
    if rank_row >= (size // args.replication):
        return

    # row_groups, col_groups = get_proc_groups(rank, size, args.replication)
    row_groups = None
    col_groups = None

    # if local_rank > 0:
    #     model = model.to(device)
    model = InteractionGNN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.nb_node_layer,
                      args.nb_edge_layer,
                      args.n_graph_iters,
                      node_features,
                      edge_features,
                      args.layernorm,
                      args.batchnorm,
                      args.hidden_activation,
                      args.output_activation,
                      args.aggr,
                      args.checkpointing,
                      args.replication,
                      impl=args.impl,
                      dataset=args.dataset)
    
    print(f"model: {model}")
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=[local_rank], find_unused_parameters=True)

    # use optimizer
    print(f"lr: {args.lr}")
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            # lr=args.lr * size,
            # lr=args.lr * math.sqrt(size),
            betas=(0.9, 0.999),
            # betas=(1 - size * (1 - 0.9), 1 - size * (1 - 0.999)),
            # eps=1e-08 / math.sqrt(size),
            eps=1e-08,
            amsgrad=True,
            weight_decay=0.01
        )
    print(f"optimizer: {optimizer}")
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=15,
                    gamma=0.9,
                )

    if args.impl == "torch":
        local_batch_size = int(args.batch_size // size)
        train_loader = ShaDowKHopSampler(trainset, 
                                            depth=args.n_layers, 
                                            num_neighbors=args.num_neighbors, 
                                            batch_size=args.batch_size, 
                                            num_workers=1,
                                            pin_memory=True,
                                            shuffle=False,
                                            drop_last=True)
        batch_start = rank * (len(train_loader) // size)
        batch_stop = min((rank + 1) * (len(train_loader) // size), len(train_loader))
        batch_count = len(train_loader)
        if rank == size - 1:
            batch_stop = len(train_loader)
        print(f"batch_start: {batch_start} batch_stop: {batch_stop}")
        print(f"len(train_loader): {len(train_loader)}")
        # train_loader = DataLoader(trainset, 
        #                             batch_size=1, 
        #                             num_workers=32,
        #                             shuffle=False,
        #                             drop_last=True)
        print(f"train_loader depth: {args.n_layers} nn: {args.num_neighbors} batch_size: {args.batch_size}")
        print(f"len(train_loader): {len(train_loader)}")
        # trainset.edge_index = trainset.adj_t.coo()
    test_loader = DataLoader(testset, batch_size=1, num_workers=1)

    row_n_bulkmb = int(args.n_bulkmb / (size / args.replication))
    if rank == size - 1:
        row_n_bulkmb = args.n_bulkmb - row_n_bulkmb * (proc_row - 1)

    rank_n_bulkmb = int(row_n_bulkmb / args.replication)
    if rank == size - 1:
        rank_n_bulkmb = row_n_bulkmb - rank_n_bulkmb * (args.replication - 1)

    print(f"global_rank: {global_rank} rank_n_bulkmb: {rank_n_bulkmb} bulkmb: {args.n_bulkmb} node_count: {node_count}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if (args.warmup != -1) and (epoch < args.warmup):
            lr_scale = min(1.0, float(epoch + 1) / args.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * args.lr # * size
        # current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch: {epoch:04}")
        if epoch >= 1:
            epoch_start = time.time()

        if args.impl == "torch":
            # for i, (batch, n_id) in enumerate(train_loader):

            batch_count = args.batchcount
            # for i in range(0, batch_count, size):
            #     batch_ids = torch.arange((i + rank) * args.batch_size, (i + rank + 1) * args.batch_size)
            for i in range(0, batch_count):
                batches_start = i * args.batch_size + rank * local_batch_size
                batches_stop = i * args.batch_size + (rank + 1) * local_batch_size
                batch_ids = torch.arange(batches_start, batches_stop)
                if epoch >= 1:
                    start_time(start_timer, timing_arg=True)
                batch = train_loader.__collate__(batch_ids)
                if epoch >= 1:
                    model.module.timings["sample"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))

                print(f"Rank: {rank:03} Epoch: {epoch:03d} Batch {i:07}")
                optimizer.zero_grad()
                if epoch >= 1:
                    start_time(start_timer, timing_arg=True)
                batch = batch.to(device)
                logits = model(batch, epoch)
                loss, pos_loss, neg_loss = model.module.loss_function(logits, batch) 

                # print(f"rank: {rank} batch {i} loss: {loss} pos_loss: {pos_loss} neg_loss: {neg_loss}", flush=True)
                # if args.wandb and rank == 0:
                #     wandb.log({'loss': loss.item(),
                #                     'pos_loss': pos_loss.item(), 
                #                     'neg_loss': neg_loss.item(), 
                #                     'epoch': epoch})
                loss.backward()
                if epoch >= 1:
                    model.module.timings["train"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))

                # if epoch >= 1:
                #     start_time(start_timer, timing_arg=True)
                # # for W in model.parameters():
                # #     if W.grad is not None:
                # #         dist.all_reduce(W.grad.data)
                # #         W.grad.data /= size
                # model.ignn_allreduce(model.named_parameters(), size, epoch)
                # if epoch >= 1:
                #     model.module.timings["allreduce"].append(stop_time(start_timer, stop_timer, barrier=True, timing_arg=True))
                optimizer.step()

        elif args.impl == "cagnet":
            batches_all = torch.arange(node_count).to(device)
            batch_count = args.batchcount

            last_iter = False
            for b in range(0, batch_count, args.n_bulkmb):
                print(f"Rank: {rank:03} Batch Bulk: {b:07}")
                if b + args.n_bulkmb > batch_count:
                    last_iter = True
                    old_nbulkmb = args.n_bulkmb
                    old_rank_nbulkmb = rank_n_bulkmb
                    args.n_bulkmb = batch_count - b
                    rank_n_bulkmb = batch_count - b
                batches_start = b * args.batch_size
                batches_stop = (b + args.n_bulkmb) * args.batch_size
                batches = batches_all[batches_start:batches_stop].view(args.n_bulkmb, args.batch_size)
                # batches_loc = torch.arange(b * args.batch_size, (b + args.n_bulkmb) * args.batch_size).view(args.n_bulkmb, args.batch_size).to(device)
                batches_loc = one5d_partition_mb(rank, size, batches, 1, args.batch_size)

                batches_indices_rows = torch.arange(batches_loc.size(0), dtype=torch.int32, device=device)
                batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
                # batches_indices_cols = batches_loc.view(-1)
                batches_indices_cols = batches_loc.reshape(-1)
                batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
                batches_values = torch.cuda.FloatTensor(batches_loc.size(1) * batches_loc.size(0), 
                                                            device=device).fill_(1.0)
                batches_loc = torch.sparse_coo_tensor(batches_indices, batches_values, 
                                                            (batches_loc.size(0), node_count))

                node_count = trainset.x.size(0)
                if args.n_darts == -1:
                    avg_degree = int(edge_count / node_count)
                    args.n_darts = []
                    avg_degree = edge_count / node_count
                    for d in range(args.n_layers):
                        dart_count = int(avg_degree * args.samp_num / avg_degree)
                        args.n_darts.append(dart_count)

                if args.replicate_graph:
                    rep_pass = 1
                else:
                    rep_pass = args.replication

                if epoch >= 1:
                    start_time(start_timer, timing_arg=True)
                local_batch_size = args.batch_size // size
                frontiers_bulk, adj_matrices_bulk = shadow_sampler(g_loc_t, 
                                                batches_loc, local_batch_size, \
                                                [args.num_neighbors] * args.n_layers, 
                                                args.n_bulkmb, \
                                                args.n_layers, args.n_darts, \
                                                rep_pass, rank, size, row_groups, \
                                                col_groups, args.timing, \
                                                model.module.timings, \
                                                args.replicate_graph, epoch)
                if epoch >= 1:
                    torch.cuda.synchronize()
                    model.module.timings["sample"].append(stop_time(start_timer, stop_timer, timing_arg=True))

                if epoch >= 1:
                    start_time(start_timer, timing_arg=True)

                # for i in range(rank_n_bulkmb):
                for i in range(args.n_bulkmb):
                    if epoch >= 1:
                        start_time(start_inner_timer, timing_arg=True)
                    adj_matrix = adj_matrices_bulk[i].to_sparse_coo()

                    frontier_cpu = frontiers_bulk[i].cpu()
                    edge_ids_cpu = adj_matrix._values().cpu()

                    if args.dataset == "physics_ex3":
                        batch = Batch(batch=frontiers_bulk[i], 
                                    edge_index=adj_matrices_bulk[i]._indices(),
                                    y=trainset.y[edge_ids_cpu],
                                    z=trainset.z[frontier_cpu],
                                    weights=trainset.weights[edge_ids_cpu],
                                    r=trainset.r[frontier_cpu],
                                    phi=trainset.phi[frontier_cpu],
                                    eta=trainset.eta[frontier_cpu],
                                    truth_map=trainset.truth_map)
                    elif args.dataset == "ctd":
                        batch = Batch(hit_id=trainset.hit_id[frontier_cpu],
                                    x=trainset.x[frontier_cpu], 
                                    y=trainset.y[edge_ids_cpu], 
                                    z=trainset.z[frontier_cpu], 
                                    edge_index=adj_matrices_bulk[i]._indices(), 
                                    truth_map=trainset.truth_map,
                                    weights=trainset.weights[edge_ids_cpu],
                                    r=trainset.r[frontier_cpu],
                                    phi=trainset.phi[frontier_cpu],
                                    eta=trainset.eta[frontier_cpu],
                                    cluster_r_1=trainset.cluster_r_1[frontier_cpu],
                                    cluster_phi_1=trainset.cluster_phi_1[frontier_cpu],
                                    cluster_z_1=trainset.cluster_z_1[frontier_cpu],
                                    cluster_eta_1=trainset.cluster_eta_1[frontier_cpu],
                                    cluster_r_2=trainset.cluster_r_2[frontier_cpu],
                                    cluster_phi_2=trainset.cluster_phi_2[frontier_cpu],
                                    cluster_z_2=trainset.cluster_z_2[frontier_cpu],
                                    cluster_eta_2=trainset.cluster_eta_2[frontier_cpu],
                                    dr=trainset.dr[edge_ids_cpu],
                                    dphi=trainset.dphi[edge_ids_cpu],
                                    dz=trainset.dz[edge_ids_cpu],
                                    deta=trainset.deta[edge_ids_cpu],
                                    phislope=trainset.phislope[edge_ids_cpu],
                                    rphislope=trainset.rphislope[edge_ids_cpu],
                                )
                    if epoch >= 1:
                        torch.cuda.synchronize()
                        model.module.timings["extract"].append(stop_time(start_inner_timer, stop_inner_timer, timing_arg=True))

                    if epoch >= 1:
                        start_time(start_inner_timer, timing_arg=True)
                    optimizer.zero_grad()
                    batch = batch.to(device)
                    if epoch >= 1:
                        torch.cuda.synchronize()
                        model.module.timings["h2d"].append(stop_time(start_inner_timer, stop_inner_timer, timing_arg=True))
                    if epoch >= 1:
                        start_time(start_inner_timer, timing_arg=True)
                    logits = model(batch, epoch)
                    if epoch >= 1:
                        torch.cuda.synchronize()
                        model.module.timings["fwd"].append(stop_time(start_inner_timer, stop_inner_timer, timing_arg=True))

                    loss, pos_loss, neg_loss = model.module.loss_function(logits, batch)
                    # print(f"batch {i} loss: {loss} pos_loss: {pos_loss} neg_loss: {neg_loss}", flush=True)
                    print(f"Rank: {rank:03} Epoch: {epoch:03d} Batch {(b + i):07}")
                    if args.wandb and rank == 0:
                        wandb.log({'loss': loss.item(),
                                        'pos_loss': pos_loss.item(), 
                                        'neg_loss': neg_loss.item(), 
                                        'epoch': epoch})
                    if epoch >= 1:
                        start_time(start_inner_timer, timing_arg=True)
                    loss.backward()
                    optimizer.step()
                    if epoch >= 1:
                        torch.cuda.synchronize()
                        model.module.timings["bwd"].append(stop_time(start_inner_timer, stop_inner_timer, timing_arg=True))
                    # if epoch >= 1:
                    #     start_time(start_inner_timer, timing_arg=True)
                    # # for W in model.parameters():
                    # #     if W.grad is not None:
                    # #         dist.all_reduce(W.grad)
                    # #         W.grad /= size
                    # # model.module.ignn_allreduce(model.named_parameters(), size, epoch)
                    # if epoch >= 1:
                    #     model.module.timings["allreduce"].append(stop_time(start_inner_timer, stop_inner_timer, timing_arg=True))

                if last_iter:
                    last_iter = False
                    args.n_bulkmb = old_nbulkmb
                    rank_n_bulkmb = old_rank_nbulkmb 
                if epoch >= 1:
                    torch.cuda.synchronize()
                    model.module.timings["train"].append(stop_time(start_timer, stop_timer, timing_arg=True))

        torch.cuda.synchronize()
        dist.barrier()
        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        print(f"Epoch: {epoch} lr: {curr_lr}", flush=True)
        if epoch >= 1:
            dur.append(time.time() - epoch_start)
            print(f"Epoch time: {(np.sum(dur) / epoch):0.03f}", flush=True)

        # if rank == 0 and epoch >= 1: # and (epoch % 5 == 0 or epoch == args.n_epochs - 1):
        if epoch >= 1: # and (epoch % 5 == 0 or epoch == args.n_epochs - 1):
            if args.timing:
                sample_dur = [x / 1000 for x in model.module.timings["sample"]]
                train_dur = [x / 1000 for x in model.module.timings["train"]]
                extract_dur = [x / 1000 for x in model.module.timings["extract"]]
                fwdbwd_dur = [x / 1000 for x in model.module.timings["fwdbwd"]]
                fwd_dur = [x / 1000 for x in model.module.timings["fwd"]]
                bwd_dur = [x / 1000 for x in model.module.timings["bwd"]]
                h2d_dur = [x / 1000 for x in model.module.timings["h2d"]]
                allreduce_dur = [x / 1000 for x in model.module.timings["allreduce"]]
                allreduce_pt1_dur = [x / 1000 for x in model.module.timings["allreduce-pt1"]]
                allreduce_pt2_dur = [x / 1000 for x in model.module.timings["allreduce-pt2"]]
                allreduce_pt3_dur = [x / 1000 for x in model.module.timings["allreduce-pt3"]]
                allreduce_pt4_dur = [x / 1000 for x in model.module.timings["allreduce-pt4"]]

                shadow_sampling_dur = [x / 1000 for x in model.module.timings["shadow-sampling"]]
                shadow_selection_dur = [x / 1000 for x in model.module.timings["shadow-selection"]]
                shadow_extraction_dur = [x / 1000 for x in model.module.timings["shadow-extraction"]]

                fwd_prelude_dur = [x / 1000 for x in model.module.timings["fwd-prelude"]]
                fwd_encoder_dur = [x / 1000 for x in model.module.timings["fwd-encoder"]]
                fwd_graphiters_dur = [x / 1000 for x in model.module.timings["fwd-graphiters"]]

                # print(f"Sample: {(np.sum(sample_dur) / epoch):0.03f} Train: {(np.sum(train_dur) / epoch):0.03f}", flush=True)
                # print(f"avg(allreduce_pt2_dur): {sum(allreduce_pt2_dur) / len(allreduce_pt2_dur)}", flush=True)
                # print(f"med(allreduce_pt2_dur): {median(allreduce_pt2_dur)}", flush=True)
                print(f"Rank: {rank} Sample: {(np.sum(sample_dur) / epoch):0.03f} Train: {(np.sum(train_dur) / epoch):0.03f} AllReduce: {(np.sum(allreduce_dur) / epoch):0.03f}", flush=True)
                print(f"Rank: {rank} AllReducePt1: {(np.sum(allreduce_pt1_dur) / epoch):0.03f} AllReducePt2: {(np.sum(allreduce_pt2_dur) / epoch):0.03f}")
                print(f"Rank: {rank} AllReducePt3: {(np.sum(allreduce_pt3_dur) / epoch):0.03f} AllReducePt4: {(np.sum(allreduce_pt4_dur) / epoch):0.03f}")
                print(f"Rank: {rank} Extract: {(np.sum(extract_dur) / epoch):0.03f} FwdBwd: {(np.sum(fwdbwd_dur) / epoch):0.03f}")
                print(f"Rank: {rank} Fwd: {(np.sum(fwd_dur) / epoch):0.03f} Bwd: {(np.sum(bwd_dur) / epoch):0.03f} H2D: {(np.sum(h2d_dur) / epoch):0.03f}")
                print(f"Rank: {rank} ShaDow-Sampling: {(np.sum(shadow_sampling_dur) / epoch):0.03f} ShaDow-Selection: {(np.sum(shadow_selection_dur) / epoch):0.03f} ShaDow-Extraction: {(np.sum(shadow_extraction_dur) / epoch):0.03f}", flush=True)
                print(f"Rank: {rank} Fwd-Prelude: {(np.sum(fwd_prelude_dur) / epoch):0.03f} Fwd-Encoder: {(np.sum(fwd_encoder_dur) / epoch):0.03f} Fwd-Graphiters: {(np.sum(fwd_graphiters_dur) / epoch):0.03f}", flush=True)

        # torch.cuda.synchronize()
        # dist.barrier()

        if not args.skip_validation:
            model.eval()
            print(f"Evaluating", flush=True)
            with torch.no_grad():
                model = model.cpu()
                result, masked_auc = model.evaluate(test_loader, epoch, curr_lr[0], \
                                                        np.sum(dur), args.wandb, rank)
                model = model.to(device)
            if args.wandb:
                if args.dataset == "ctd" and rank == 0:
                    wandb.log({'signal_effiency': result,
                                'time': np.sum(dur)})
                elif rank == 0:
                    wandb.log({'full_auc': result,
                                'time': np.sum(dur)})

            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()

    test_loader = DataLoader(testset, batch_size=1, num_workers=1)
    model.eval()
    with torch.no_grad():
        if rank == 0:
            print(f"Evaluating", flush=True)
            # model = model.cpu()
            # model = model.cuda()
            result, masked_auc = model.evaluate(test_loader, args.n_epochs, 
                                                            0.0, 0.0, args.wandb, 0)
        # model = model.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IGNN')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--sample-method", type=str, default="shadow",
                        help="sampling algorithm for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=4,
                        help="gpus per node")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of vertices in minibatch")
    parser.add_argument("--num-neighbors", type=int, default=1,
                        help="number of neigbors per vertex of minibatch")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggr", type=str, default="mean",
                        help="Aggregator type: mean/sum")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument('--normalize', action="store_true",
                            help='normalize adjacency matrix')
    parser.add_argument('--partitioning', default='ONE5D', type=str,
                            help='partitioning strategy to use')
    parser.add_argument('--replication', default=1, type=int,
                            help='partitioning strategy to use')
    parser.add_argument('--n-bulkmb', default=1, type=int,
                            help='number of minibatches to sample in bulk')
    parser.add_argument('--bulk-batch-fetch', default=1, type=int,
                            help='number of minibatches to fetch features for in bulk')
    parser.add_argument('--n-darts', default=-1, type=int,
                            help='number of darts to throw per minibatch in LADIES \
                                    sampling')
    parser.add_argument('--semibulk', default=128, type=int,
                            help='number of batches to column extract from in bulk')
    parser.add_argument('--timing', action="store_true",
                            help='whether to turn on timers')
    parser.add_argument('--baseline', action="store_true",
                            help='whether to avoid col selection for baseline \
                                    comparison')
    parser.add_argument('--replicate-graph', action="store_true",
                            help='replicate adjacency matrix on each device')
    parser.add_argument("--nb-node-layer", type=int, default=2,
                        help="number of hidden node MLP layers")
    parser.add_argument("--nb-edge-layer", type=int, default=2,
                        help="number of hidden edge MLP layers")
    parser.add_argument("--n-graph-iters", type=int, default=8,
                        help="number of message passing iterations")
    parser.add_argument('--batchnorm', action="store_true",
                            help='use batchnorm in mlps')
    parser.add_argument('--layernorm', action="store_true",
                            help='use batchnorm in mlps')
    parser.add_argument('--hidden-activation', choices=["SiLU", "ReLU", "Tanh"],
                            help='hidden activation in mlps')
    parser.add_argument('--output-activation', choices=["SiLU", "ReLU", "Tanh"],
                            help='output activation in mlps')
    parser.add_argument('--impl', default='cagnet', type=str,
                            help='sampling implementation to use (torch/cagnet)')
    parser.add_argument('--wandb', action="store_true",
                            help='use wandb logging')
    parser.add_argument('--checkpointing', action="store_true",
                            help='use checkpointing')
    parser.add_argument("--warmup", type=int, default=5,
                        help="number of warmup iterations for learning rate")
    parser.add_argument("--batchcount", type=int, default=5,
                        help="number of warmup iterations for learning rate")
    parser.add_argument('--skip-validation', action="store_true",
                            help='skip validation step when expensive')
    args = parser.parse_args()
    args.samp_num = args.num_neighbors

    print(args)

    if args.dataset == "cora" or args.dataset == "reddit":
        if args.dataset == "cora":
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset == "reddit":
            dataset = Reddit(path)

        data = dataset[0]
        data = data.to(device)
        # data.x.requires_grad = True
        inputs = data.x.to(device)
        # inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        adj_matrix = edge_index
    elif args.dataset == "physics_ex3":
        print(f"Loading coo...", flush=True)
        input_dir = "/pscratch/sd/a/alokt/data_dir/Example_3/metric_learning/"
        with open(f"gnn_train_{args.dataset}.yaml") as stream:
            hparams = yaml.safe_load(stream)

        print(f"hparams: {hparams}", flush=True)

        dataset = GraphDataset(input_dir, "trainset", 80, "fit", hparams)

        print(f"dataset: {dataset}", flush=True)
        trainset = []
        for data in dataset:
            data_obj = Data(hit_id=data["hit_id"],
                                x=data["x"], 
                                y=data["y"], 
                                z=data["z"], 
                                r=data["r"], 
                                phi=data["phi"], 
                                eta=data["eta"], 
                                edge_index=data["edge_index"], 
                                truth_map=data["truth_map"],
                                weights=data["weights"])

            # data_obj = dataset.preprocess_event(data_obj)
            trainset.append(data_obj)
        trainset = Batch.from_data_list(trainset)
        # trainset = trainset.to(device)
        trainset.x.share_memory_()
        trainset.y.share_memory_()
        trainset.z.share_memory_()
        trainset.hit_id.share_memory_()
        trainset.edge_index.share_memory_()
        trainset.truth_map.share_memory_()
        trainset.weights.share_memory_()
        trainset.r.share_memory_()
        trainset.phi.share_memory_()
        trainset.eta.share_memory_()
        print(f"trainset: {trainset}", flush=True)

        node_count = torch.max(trainset.edge_index) + 1
        edge_count = trainset.edge_index.size(1)
        node_features = ["r", "phi", "z"]
        edge_features = []

        testset = GraphDataset(input_dir, "testset", 10, "fit", hparams)
        print(f"testset: {testset}", flush=True)
        num_features = len(node_features)
        num_classes = 2

        g_loc_indices = trainset.edge_index
        g_loc_values = torch.arange(g_loc_indices.size(1), dtype=torch.int64)
        g_loc = torch.sparse_coo_tensor(g_loc_indices, g_loc_values)
        g_loc = g_loc.to_sparse_csr()
        g_loc_t = g_loc.t().to_sparse_csr()
        print(f"g_loc: {g_loc}", flush=True)

    elif args.dataset == "ctd":
        print(f"Loading coo...", flush=True)
        # input_dir = "/global/cfs/cdirs/m4439/CTD2023_results/module_map/"
        input_dir = "/global/cfs/cdirs/m4439/REL2024/metric_learning_7_30/"
        with open(f"gnn_train_{args.dataset}.yaml") as stream:
            hparams = yaml.safe_load(stream)

        print(f"hparams: {hparams}", flush=True)
        
        # trainset = GraphDataset(input_dir, "trainset", 4053, "fit", hparams)
        dataset = GraphDataset(input_dir, "trainset", 80, "fit", hparams)
        trainset = []
        for data in dataset:
            if data is None:
                continue
            data_obj = Data(hit_id=data["hit_id"],
                                x=data["x"], 
                                y=data["y"], 
                                z=data["z"], 
                                edge_index=data["edge_index"], 
                                # edge_index=edge_index, 
                                truth_map=data["truth_map"],
                                weights=data["weights"],
                                r=data["r"],
                                phi=data["phi"],
                                eta=data["eta"],
                                cluster_r_1=data["cluster_r_1"],
                                cluster_phi_1=data["cluster_phi_1"],
                                cluster_z_1=data["cluster_z_1"],
                                cluster_eta_1=data["cluster_eta_1"],
                                cluster_r_2=data["cluster_r_2"],
                                cluster_phi_2=data["cluster_phi_2"],
                                cluster_z_2=data["cluster_z_2"],
                                cluster_eta_2=data["cluster_eta_2"],
                                dr=data["dr"],
                                dphi=data["dphi"],
                                dz=data["dz"],
                                deta=data["deta"],
                                phislope=data["phislope"],
                                rphislope=data["rphislope"],
                            )

            trainset.append(data_obj)
        trainset = Batch.from_data_list(trainset)
        trainset.x.share_memory_()
        trainset.y.share_memory_()
        trainset.z.share_memory_()
        trainset.hit_id.share_memory_()
        trainset.edge_index.share_memory_()
        trainset.truth_map.share_memory_()
        trainset.weights.share_memory_()
        trainset.r.share_memory_()
        trainset.phi.share_memory_()
        trainset.eta.share_memory_()
        trainset.cluster_r_1.share_memory_()
        trainset.cluster_phi_1.share_memory_()
        trainset.cluster_z_1.share_memory_()
        trainset.cluster_eta_1.share_memory_()
        trainset.cluster_r_2.share_memory_()
        trainset.cluster_phi_2.share_memory_()
        trainset.cluster_z_2.share_memory_()
        trainset.cluster_eta_2.share_memory_()
        trainset.dr.share_memory_()
        trainset.dphi.share_memory_()
        trainset.dz.share_memory_()
        trainset.deta.share_memory_()
        trainset.phislope.share_memory_()
        trainset.rphislope.share_memory_()
        trainset.batch.share_memory_()
        trainset.ptr.share_memory_()
        node_count = torch.max(trainset.edge_index) + 1
        edge_count = trainset.edge_index.size(1)
        node_features = ["r", "phi", "z", "eta", "cluster_r_1", "cluster_phi_1", 
                            "cluster_z_1", "cluster_eta_1", "cluster_r_2", 
                            "cluster_phi_2", "cluster_z_2", "cluster_eta_2"]
        edge_features = ["dr", "dphi", "dz", "deta", "phislope", "rphislope"]

        g_loc_indices = trainset.edge_index
        g_loc_values = torch.arange(g_loc_indices.size(1), dtype=torch.int64)
        g_loc = torch.sparse_coo_tensor(g_loc_indices, g_loc_values)
        g_loc = g_loc.to_sparse_csr()
        g_loc_t = g_loc.t().to_sparse_csr()
        # trainset = trainset.to(device)
        print(f"dataset: {dataset}", flush=True)
        print(f"trainset: {trainset}", flush=True)
    
        # input_dir_test = "/global/cfs/cdirs/m4439/CTD2023_results/module_map/"
        input_dir_test = "/global/cfs/cdirs/m4439/REL2024/metric_learning_7_30/"
        testset = GraphDataset(input_dir_test, "valset", 10, "test", hparams)
        print(f"testset: {testset}", flush=True)

        num_features = len(node_features)
        num_classes = 2


    if args.wandb and int(os.environ["SLURM_PROCID"]) == 0:
        wandb.init(project="exatrkx")

    # torch.set_num_threads(1)
    mp.set_start_method("spawn", force=True)
    mp.spawn(main,
             args=(args, trainset, testset, 
                    g_loc.crow_indices(), g_loc.col_indices(), g_loc.values(), 
                    g_loc_t.crow_indices(), g_loc_t.col_indices(), g_loc_t.values(), 
                    node_count, edge_count, node_features, edge_features, 
                    num_features, num_classes),
             nprocs=args.gpu,
             join=True)
    # processes = []
    # for i in range(args.gpu):
    #     p = Process(target=main,
    #             args=(i, args, model, trainset, testset, 
    #                 g_loc.crow_indices(), g_loc.col_indices(), g_loc.values(), 
    #                 g_loc_t.crow_indices(), g_loc_t.col_indices(), g_loc_t.values(), 
    #                 node_count, edge_count, node_features, edge_features, 
    #                 num_features, num_classes))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

