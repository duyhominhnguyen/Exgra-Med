# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import timm
import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, size="base"):
        super(ViT, self).__init__()
        self.model = timm.create_model(f"vit_{size}_patch16_224", num_classes=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = self.model.pre_logits(x)
        x_avg = self.avgpool(x)
        x_avg = torch.squeeze(x_avg)

        return x, x_avg


def vit_small_224(**kwargs):
    return ViT(size="small"), 384


def vit_base_224(**kwargs):
    return ViT(size="base"), 768


def vit_large_224(**kwargs):
    return ViT(size="large"), 1024
