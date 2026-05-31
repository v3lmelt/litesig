"""Space-to-depth convolution modules used in the Hybrid Siamese network.

This module mirrors the HTCSigNet idea of replacing stride pooling with a
space-to-depth (pixel unshuffle) operation followed by strided convolution.
The goal is to keep fine-grained pen-stroke cues while gradually expanding
receptive fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class SPDConvConfig:
    """Configuration holder to ease ablation studies."""

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    reduction: int = 2
    use_bn: bool = True
    activation: str = "relu"

    def space_to_depth_scale(self) -> int:
        if self.reduction not in (1, 2, 4):
            raise ValueError("reduction must be 1, 2, or 4 for pixel unshuffle")
        return self.reduction


class SPDConv(nn.Module):
    """Space-to-depth + conv block.

    This block performs a pixel unshuffle (space-to-depth) to move spatial
    resolution into the channel dimension, followed by a stride-1 convolution.
    It mimics the SPD-Conv module from HTCSigNet.
    """

    def __init__(self, cfg: SPDConvConfig):
        super().__init__()
        self.cfg = cfg
        scale = cfg.space_to_depth_scale()
        self.pixel_unshuffle = nn.PixelUnshuffle(scale) if scale > 1 else nn.Identity()
        in_ch = cfg.in_channels * (scale**2)
        padding = (cfg.kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, cfg.out_channels, cfg.kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(cfg.out_channels) if cfg.use_bn else nn.Identity()
        self.act = _build_activation(cfg.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_unshuffle(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")
