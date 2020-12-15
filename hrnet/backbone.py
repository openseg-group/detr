# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from models.position_encoding import build_position_encoding

from models.backbone import FrozenBatchNorm2d
from .hrnet import build_hrnet

import pdb
import os

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()

        # if int(os.environ.get("freeze_stem_stage1", 0)):
        #     for name, parameter in backbone.named_parameters():
        #         if not train_backbone or ('stage2' not in name and 'stage3' not in name and 'stage4' not in name \
        #             and 'transition1' not in name and 'transition2' not in name and 'transition3' not in name):
        #             parameter.requires_grad_(False)

        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """HRNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool):
        backbone = build_hrnet(name, pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # [48, 96, 192, 384]
        # [32, 64, 128, 256]
        # [18, 36, 72, 144]
        num_channels = 8 * int(name[-2:])
        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
