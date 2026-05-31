"""Feature fusion head that concatenates branch outputs."""
from __future__ import annotations

from typing import List

import torch
from torch import nn


class FusionHead(nn.Module):
    """Normalizes and projects concatenated multi-branch embeddings."""

    def __init__(self, input_dims: List[int], embed_dim: int = 512, dropout: float = 0.1, identity: bool = False):
        super().__init__()
        total_dim = sum(input_dims)
        self.identity = identity
        if identity:
                                                            
            self.norm = None
            self.mlp = None
        else:
            self.norm = nn.LayerNorm(total_dim)
            self.mlp = nn.Sequential(
                nn.Linear(total_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(features, dim=-1)
        if self.identity:
            return _l2_normalize(fused)
        fused = self.norm(fused)
        fused = self.mlp(fused)
        return _l2_normalize(fused)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
