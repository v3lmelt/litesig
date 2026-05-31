"""Spatial-Channel Reconstruction Convolution (SCConv) module.

SCConv follows the HTCSigNet style SRU/CRU structure to suppress redundant
responses while amplifying subtle stroke deviations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SCConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    groups: int = 1
    use_bn: bool = True
    reduction_ratio: int = 2
    activation: str = "relu"


class SCConv(nn.Module):
    """Simplified SCConv that still captures SRU + CRU behavior."""

    def __init__(self, cfg: SCConvConfig):
        super().__init__()
        padding = (cfg.kernel_size - 1) // 2
        self.conv_sru = nn.Conv2d(
            cfg.in_channels,
            cfg.out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=padding,
            groups=cfg.groups,
            bias=False,
        )
        self.conv_cru = nn.Conv2d(
            cfg.out_channels,
            cfg.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(cfg.out_channels) if cfg.use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(cfg.out_channels) if cfg.use_bn else nn.Identity()

        reduced = max(1, cfg.out_channels // cfg.reduction_ratio)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cfg.out_channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, cfg.out_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.act = _build_activation(cfg.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_sru(x)
        y = self.bn1(y)
        y = self.act(y)
        gate = self.gate(y)
        y = y * gate
        y = self.conv_cru(y)
        y = self.bn2(y)
        return self.act(y)


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")
