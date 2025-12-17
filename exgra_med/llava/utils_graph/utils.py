# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math

import torch
import pickle
import argparse
import numpy as np
from loguru import logger

# from logging import getLogger

from torch import nn
import torch.distributed as dist
from itertools import combinations as comb


# code borrowed from  paper ECCV 2020
def lexico_iter(lex):
    return comb(lex, 2)


# code borrowed from SwAV (NIPS 2020) paper
def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path,
        map_location="cuda:"
        + str(torch.distributed.get_rank() % torch.cuda.device_count()),
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def round_log(key, log, item=True, iters=1):
    if item:
        log = log.item()
    if "top" in key:
        return round(100 * (log / iters), 4)
    return round(log / iters, 6)


def checkpoint(args, epoch, step, model, optimizer, name=""):
    if args.rank != 0 or epoch % args.checkpoint_freq != 0:
        return
    state = dict(
        epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict()
    )
    save_name = f"model_{name}.pth" if len(name) > 0 else "model.pth"
    torch.save(state, args.exp_dir / save_name)


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


def learning_schedule(
    global_step, batch_size, base_lr, end_lr_ratio, total_steps, warmup_steps
):
    scaled_lr = base_lr * batch_size / 256.0
    end_lr = scaled_lr * end_lr_ratio
    if global_step < warmup_steps:
        learning_rate = (
            global_step / warmup_steps * scaled_lr if warmup_steps > 0 else scaled_lr
        )
        return learning_rate
    else:
        return cosine_decay(
            global_step - warmup_steps, total_steps - warmup_steps, scaled_lr, end_lr
        )


def cosine_decay(global_step, max_steps, initial_value, end_value):
    global_step = min(global_step, max_steps)
    cosine_decay_value = 0.5 * (1 + math.cos(math.pi * global_step / max_steps))
    return (initial_value - end_value) * cosine_decay_value + end_value


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x.contiguous())
    return torch.cat(x_list, dim=0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res


def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x


def MLP(mlp, embedding, norm_layer):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if norm_layer == "batch_norm":
            layers.append(nn.BatchNorm1d(f[i + 1]))
        elif norm_layer == "layer_norm":
            layers.append(nn.LayerNorm(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    #     layers.append(nn.LayerNorm(f[-1]))
    return nn.Sequential(*layers)
