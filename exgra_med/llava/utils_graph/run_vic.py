# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import os

# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
# os.environ["OMP_NUM_THREADS"] = "1"
# torch.set_num_threads(1)
import sys
import time
import json
import apex
import argparse
import numpy as np
from pathlib import Path
from logging import getLogger
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
from torch import optim
import torch.nn.parallel
from torchvision import models
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset

import torch_geometric
from torch_geometric import loader
from torch_geometric.data import Data
from torch_geometric.nn import SplineConv, EGConv, GraphConv, GATConv, GCNConv

from utils.config import cfg
from utils.utils import lexico_iter, update_params_from_cmdline

cfg = update_params_from_cmdline(default_params=cfg)

from lpmp_py import GraphMatchingModule
from hoang_BBGM.dataset import SSL_Dataset
from hoang_BBGM.build_graph import build_graphs, build_graphs_main, build_graphs_main_1

# import fuck
import utils
from datasets import build_loader
from resnet import resnet50
from optimizers import build_optimizer
from distributed import init_distributed_mode

sys.path.append("/home/caduser/HDD/hoang_graph_matching/")
from swav.src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
)

logger = getLogger()


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Pretraining with VICRegL", add_help=False
    )
    #########################
    #### data parameters ####
    #########################
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/caduser/HDD/hoang_graph_matching/log/",
        help="path to tensorboard folder",
    )
    parser.add_argument(
        "--tensorboard_name",
        type=str,
        default="bbgm_vic",
        help="name of tensorboard folder",
    )
    parser.add_argument("--checkpoint_name", type=str, default="bbgm_vic.pth.tar")
    #########################
    #### optim parameters ###
    #########################
    parser.add_argument(
        "--epochs", default=100, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size",
        default=800,
        type=int,
        help="batch size per gpu, i.e. how many unique instances per gpu",
    )
    parser.add_argument(
        "--num_image_per_graph",
        default=10,
        type=int,
        help="number of sample images per graph",
    )
    parser.add_argument(
        "--num_neighbors",
        default=5,
        type=int,
        help="number of neighbors in KNN algorithm",
    )
    #########################
    #### dist parameters ###
    #########################
    # Checkpoints and Logs
    #     parser.add_argument("--exp-dir", type=Path, default= "./")
    #     parser.add_argument("--log-tensors-interval", type=int, default=60)
    parser.add_argument("--checkpoint-freq", type=int, default=1)
    # Data
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    parser.add_argument("--dataset_from_numpy", action="store_true")
    parser.add_argument("--size-crops", type=int, nargs="+", default=[224, 96])
    parser.add_argument("--num-crops", type=int, nargs="+", default=[2, 2])
    parser.add_argument("--min_scale_crops", type=float, nargs="+", default=[0.4, 0.08])
    parser.add_argument("--max_scale_crops", type=float, nargs="+", default=[1, 0.4])
    parser.add_argument("--no-flip-grid", type=int, default=1)

    # Model
    #     parser.add_argument("--arch", type=str, default="convnext_small")
    #     parser.add_argument("--drop-path-rate", type=float, default=0.1)
    #     parser.add_argument("--layer-scale-init-value", type=float, default=0.0)
    parser.add_argument("--mlp", default="1024-1024-1024")
    parser.add_argument("--maps-mlp", default="512-512-512")

    # Loss Function
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--num_matches",
        type=int,
        nargs="+",
        default=[20, 4],
        help="Number of spatial matches in a feature map",
    )
    parser.add_argument("--l2_all_matches", type=int, default=1)
    parser.add_argument("--inv-coeff", type=float, default=25.0)
    parser.add_argument("--var-coeff", type=float, default=25.0)
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--fast-vc-reg", type=int, default=0)

    # Optimization
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--base-lr", type=float, default=0.0005)
    parser.add_argument("--end-lr-ratio", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    # Evaluation
    #     parser.add_argument("--val-batch-size", type=int, default=-1)
    #     parser.add_argument("--evaluate", action="store_true")
    #     parser.add_argument("--evaluate-only", action="store_true")
    #     parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--maps-lr-ratio", type=float, default=0.1)
    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument(
        "--num_workers", default=10, type=int, help="number of data loading workers"
    )
    #     parser.add_argument("--checkpoint_freq", type=int, default=25,
    #                         help="Save the model periodically")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument(
        "--syncbn_process_group_size",
        type=int,
        default=8,
        help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""",
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default="./",
        help="experiment dump path for checkpoints and log",
    )
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    # Distributed
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""",
    )
    #     parser.add_argument("--local_rank", default=0, type=int,
    #                         help="this argument is not used and should be ignored")
    return parser


def main(args):

    torch.backends.cudnn.benchmark = True
    assert args.num_image_per_graph <= args.batch_size
    writer = SummaryWriter(log_dir=args.log_dir + args.tensorboard_name)
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    gpu = torch.device(args.device)
    train_loader, train_sampler = build_loader(args, is_train=True)

    logger.info("Building data done.")
    logger.info(f"Len of train loader : {len(train_loader)}")

    model = VICRegL(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    logger.info(model)
    logger.info("Building model done.")
    optimizer = build_optimizer(args, model)
    logger.info("Building optimizer done.")
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, args.checkpoint_name),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        #         amp=None,
    )
    start_epoch = to_restore["epoch"]

    #     start_epoch = 0
    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    for epoch in range(start_epoch, args.epochs):
        logger.info("============ Starting epoch %i ... ============" % epoch)
        train_sampler.set_epoch(epoch)
        for step, inputs in enumerate(train_loader):
            end = time.time()
            lr = utils.learning_schedule(
                global_step=step,
                batch_size=args.batch_size,
                base_lr=args.base_lr,
                end_lr_ratio=args.end_lr_ratio,
                total_steps=args.epochs * len(train_loader.dataset) // args.batch_size,
                warmup_steps=args.warmup_epochs
                * len(train_loader.dataset)
                // args.batch_size,
            )
            for g in optimizer.param_groups:
                if "__MAPS_TOKEN__" in g.keys():
                    g["lr"] = lr * args.maps_lr_ratio
                else:
                    g["lr"] = lr

            optimizer.zero_grad()
            if args.fp16:
                with torch.cuda.amp.autocast():
                    loss, global_loss, local_loss, classif_loss = model.forward(
                        make_inputs(inputs, gpu)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, global_loss, local_loss, classif_loss = model.forward(
                    make_inputs(inputs, gpu)
                )
                loss.backward()
                optimizer.step()
            # ============ misc ... ============
            losses.update(loss.item(), inputs[0][0][0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and step % 50 == 0:
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        step,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=lr,
                    )
                )
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, args.checkpoint_name),
            )


def make_inputs(inputs, gpu):
    if isinstance(inputs[0][1][1], list):
        (val_view, (views, locations)), labels = inputs
        return dict(
            val_view=val_view.cuda(gpu, non_blocking=True),
            views=[view.cuda(gpu, non_blocking=True) for view in views],
            locations=[location.cuda(gpu, non_blocking=True) for location in locations],
            labels=labels.cuda(gpu, non_blocking=True),
        )
    (views, locations), labels = inputs
    return dict(
        views=[view.cuda(gpu, non_blocking=True) for view in views],
        locations=[location.cuda(gpu, non_blocking=True) for location in locations],
        labels=labels.cuda(gpu, non_blocking=True),
    )


class VICRegL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = int(args.mlp.split("-")[-1])

        self.backbone, self.representation_dim = resnet50(zero_init_residual=True)
        # representation dim of resnet 50 : 2048
        self.message_pass_node_features = GDPModel1(num_features=1024, hidden_size=1024)
        self.build_edge_features_from_node_features = (
            SiameseNodeFeaturesToEdgeFeatures()
        )
        self.vertex_affinity = InnerProductWithWeightsAffinity()
        self.edge_affinity = InnerProductWithWeightsAffinity()
        self.num_image_per_graph = args.num_image_per_graph
        self.num_neighbors = args.num_neighbors

        self.hamming_loss = HammingLoss()

        if self.args.alpha < 1.0:  # local
            self.maps_projector = utils.MLP(args.maps_mlp, self.representation_dim)
            # 512

        if self.args.alpha > 0.0:  # global
            self.projector = utils.MLP(args.mlp, self.representation_dim)
            # 1024
        self.classifier = nn.Linear(self.representation_dim, self.args.num_classes)

    def _vicreg_loss(self, x, y):
        repr_loss = self.args.inv_coeff * F.mse_loss(x, y)

        x = utils.gather_center(x)
        y = utils.gather_center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.args.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.args.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _l2_distance(self, maps_1, maps_2):

        maps_1 = torch.flatten(maps_1, start_dim=1)
        maps_2 = torch.flatten(maps_2, start_dim=1)

        return torch.cdist(maps_1, maps_2)

    def _local_loss(self, maps_1, maps_2, location_1, location_2):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        # L2 distance based bacthing
        if self.args.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.args.num_matches

        maps_1_filtered_l2, maps_1_nn_l2 = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered_l2, maps_2_nn_l2 = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )
        # Location based matching
        location_1 = location_1.flatten(1, 2)
        location_2 = location_2.flatten(1, 2)
        maps_1_filtered_location, maps_1_nn_location = neirest_neighbores_on_location(
            location_1,
            location_2,
            maps_1,
            maps_2,
            num_matches=self.args.num_matches[0],
        )
        maps_2_filtered_location, maps_2_nn_location = neirest_neighbores_on_location(
            location_2,
            location_1,
            maps_2,
            maps_1,
            num_matches=self.args.num_matches[1],
        )
        bs = maps_1.shape[0]
        device = maps_1.device
        num_graphs, redundant_images = divmod(bs, self.num_image_per_graph)
        perm_matrix = (
            torch.diag(torch.ones(self.num_image_per_graph), 0).unsqueeze(0).to(device)
        )
        perm_matrix_1 = (
            torch.diag(torch.ones(redundant_images), 0).unsqueeze(0).to(device)
        )

        prediction_list = []
        target_list = []
        losses = 0.0
        for index in range(num_graphs):
            sub_node_1 = maps_1[
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            sub_node_2 = maps_2[
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            # build graph
            orig_graph = self.build_graph(sub_node_1, sub_node_2)
            # cal local feature to c^v
            sub_maps_1_filtered_l2 = maps_1_filtered_l2[
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            sub_maps_2_nn_l2 = maps_2_filtered_l2[
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            sub_maps_1_filtered_location = maps_1_filtered_location[
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            sub_maps_2_nn_location = maps_2_filtered_location[
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            unary_costs_list = [
                self.vertex_affinity(g_1.x, g_2.x)
                for (g_1, g_2) in lexico_iter(orig_graph)
            ]
            unary_costs_list = [-x for x in unary_costs_list]
            # add 1 to diagonal and l2 distance of location feature
            unary_costs_list[0] += (
                perm_matrix[0]
                + self._l2_distance(sub_maps_1_filtered_l2, sub_maps_2_filtered_l2)
                + self._l2_distance(
                    sub_maps_1_filtered_location, sub_maps_2_filtered_location
                )
            )
            quadratic_costs_list = [
                self.edge_affinity(g_1.edge_attr, g_2.edge_attr)
                for (g_1, g_2) in lexico_iter(orig_graph)
            ]
            # Aimilarities to costs
            quadratic_costs_list = [-0.5 * x for x in quadratic_costs_list]
            gm_solvers = GraphMatchingModule(
                [orig_graph[0].edge_index],
                [orig_graph[1].edge_index],
                [self.num_image_per_graph],
                [self.num_image_per_graph],
                cfg.BB_GM.lambda_val,
                cfg.BB_GM.solver_params,
            )
            matching = gm_solvers(unary_costs_list, quadratic_costs_list)
            losses += self.hamming_loss(matching, perm_matrix)
        return losses / num_graphs

    def local_loss(self, maps_embedding, locations):
        num_views = len(maps_embedding)
        cur_loss = 0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                loss = self._local_loss(
                    maps_embedding[i],
                    maps_embedding[j],
                    locations[i],
                    locations[j],
                )
                cur_loss += loss
                iter_ += 1

        cur_loss /= iter_

        return cur_loss

    def build_graph(self, global_map1, global_map2):
        map_len = global_map1.shape[0]
        edge_indice1, edge_feature1 = build_graphs_main_1(
            global_map1, n_neighbors=self.num_neighbors, num_workers=1
        )
        edge_indice2, edge_feature2 = build_graphs_main_1(
            global_map2, n_neighbors=self.num_neighbors, num_workers=1
        )
        data_graph1 = Data(
            x=global_map1, edge_index=edge_indice1, edge_attr=edge_feature1
        )
        data_graph2 = Data(
            x=global_map2, edge_index=edge_indice2, edge_attr=edge_feature2
        )
        graph_loader = loader.DataLoader([data_graph1, data_graph2], batch_size=2)
        graph = self.message_pass_node_features(next(iter(graph_loader)))
        end = time.time()
        orig_graph = self.build_edge_features_from_node_features(graph)
        return orig_graph

    def loss_calculate(self, embedding):
        """
        data (dictionary) : must contain "view1" and "view2" elements meaning 2 augmentation views.
        """
        bs = embedding[0].shape[0]
        device = embedding[0].device
        num_graphs, redundant_images = divmod(bs, self.num_image_per_graph)
        perm_matrix = (
            torch.diag(torch.ones(self.num_image_per_graph), 0).unsqueeze(0).to(device)
        )
        perm_matrix_1 = (
            torch.diag(torch.ones(redundant_images), 0).unsqueeze(0).to(device)
        )
        prediction_list = []
        target_list = []
        loss = 0.0
        save_time = 0
        for index in range(num_graphs):

            start = time.time()
            sub_node_view1 = embedding[0][
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            sub_node_view2 = embedding[1][
                index
                * self.num_image_per_graph : (index + 1)
                * self.num_image_per_graph
            ].float()
            orig_graph = self.global_loss_BBGM_(sub_node_view1, sub_node_view2)
            #  ================================================================== #
            start = time.time()
            unary_costs_list = [
                self.vertex_affinity(g_1.x, g_2.x)
                for (g_1, g_2) in lexico_iter(orig_graph)
            ]
            unary_costs_list = [-x for x in unary_costs_list]
            unary_costs_list[0] += perm_matrix[0]
            # logger.info(f'Time to compute unary cost functions  {end - start}')
            quadratic_costs_list = [
                self.edge_affinity(g_1.edge_attr, g_2.edge_attr)
                for (g_1, g_2) in lexico_iter(orig_graph)
            ]
            # Aimilarities to costs
            quadratic_costs_list = [-0.5 * x for x in quadratic_costs_list]
            gm_solvers = GraphMatchingModule(
                [orig_graph[0].edge_index],
                [orig_graph[1].edge_index],
                [self.num_image_per_graph],
                [self.num_image_per_graph],
                cfg.BB_GM.lambda_val,
                cfg.BB_GM.solver_params,
            )
            matching = gm_solvers(unary_costs_list, quadratic_costs_list)
            prediction_list.append(matching)
            target_list.append(perm_matrix)
            start = time.time()
            loss += self.hamming_loss(matching, perm_matrix)
            end = time.time()
        loss /= num_graphs
        return loss

    def compute_metrics(self, outputs):
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                utils.off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = utils.batch_all_gather(outputs["representation"][0])
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.args.alpha > 0.0:
            embedding = utils.batch_all_gather(outputs["embedding"][0])
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            return dict(stdr=stdrepr, stde=stdemb, corr=corr, core=core)
        return dict(stdr=stdrepr, corr=corr)

    def forward_networks(self, inputs, is_val=False):
        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": [],
            "logits": [],
        }
        for x in inputs["views"]:
            maps, representation = self.backbone(x)
            outputs["representation"].append(representation)

            # global
            if self.args.alpha > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

            # local
            if self.args.alpha < 1.0:
                maps_embedding = self.maps_projector(maps)
                outputs["maps_embedding"].append(maps_embedding)

            logits = self.classifier(representation.detach())
            outputs["logits"].append(logits)
        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        if backbone_only:
            maps, _ = self.backbone(inputs)
            return maps

        outputs = self.forward_networks(inputs, is_val)
        with torch.no_grad():
            logs = self.compute_metrics(outputs)
        loss = 0.0

        if self.args.alpha < 1.0:
            loss = self.local_loss(outputs["maps_embedding"], inputs["locations"])

        # Online classification
        #         labels = inputs["labels"]
        #         classif_loss = F.cross_entropy(outputs["logits"][0], labels)
        #         acc1, acc5 = utils.accuracy(outputs["logits"][0], labels, topk=(1, 5))
        #         loss = loss + classif_loss
        #         logs.update(dict(cls_l=classif_loss, top1=acc1, top5=acc5, l=loss))

        return loss


class GDPModel1(torch.nn.Module):
    def __init__(self, num_features=512, hidden_size=512, edge_dim=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.edge_dim = edge_dim
        convs = [
            GraphConv(self.num_features, self.hidden_size),
            GraphConv(self.hidden_size, self.hidden_size),
        ]

        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)  # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        data.x = x
        return data


class SiameseNodeFeaturesToEdgeFeatures(torch.nn.Module):
    def __init__(self):
        super(SiameseNodeFeaturesToEdgeFeatures, self).__init__()

    #         self.num_edge_features = total_num_nodes
    def forward(self, graph):
        orig_graphs = graph.to_data_list()
        orig_graphs = [self.vertex_attr_to_edge_attr(graph) for graph in orig_graphs]
        return orig_graphs

    def vertex_attr_to_edge_attr(self, graph):
        """Assigns the difference of node features to each edge"""
        flat_edges = graph.edge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        new_edge_attrs = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        graph.edge_attr = new_edge_attrs
        return graph


class InnerProductWithWeightsAffinity(nn.Module):
    def __init__(self):
        super(InnerProductWithWeightsAffinity, self).__init__()

    def forward(self, X, Y):
        res = torch.matmul(X, Y.transpose(0, 1))
        res = torch.nn.functional.softplus(res) - 0.5
        return res


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def neirest_neighbores_on_location(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Pretraining with VICRegL", parents=[get_arguments()]
    )
    args = parser.parse_args()
    main(args)
