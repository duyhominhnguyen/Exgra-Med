# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import timm
import torch
import torch.nn as nn


class Swin_Transformer(nn.Module):
    def __init__(self, size="base"):
        super(Swin_Transformer, self).__init__()
        self.model = timm.create_model(f"swin_{size}_patch4_window7_224", num_classes=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        x = self.model.norm(x)
        x_avg = self.avgpool(x)
        x_avg = self.model.head(x_avg)
        x_avg = torch.squeeze(x_avg)
        return x, x_avg


def swin_small_224(**kwargs):
    return Swin_Transformer(size="small"), 768


def swin_base_224(**kwargs):
    return Swin_Transformer(size="base"), 1024


def swin_large_224(**kwargs):
    return Swin_Transformer(size="large"), 1536
