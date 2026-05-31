"""Siamese similarity head that consumes two embeddings."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LightSiameseHead(nn.Module):
    """Lightweight Siamese similarity head optimized for ultra-light/micro models.
    
    Design principles:
    - Target: ~0.3-0.5M parameters (vs ~3M for SiameseHead)
    - Simple but effective: |diff|, prod, concat → lightweight MLP
    - No SE-net, no multi-scale projections, no bilinear interaction
    - Single-scale difference weighting for key discriminative features
    
    Parameter breakdown (embed_dim=512, hidden_dim=256):
    - diff_gate: 512×64 + 64×512 = 65K
    - fc1: 1536×256 = 393K  
    - fc2: 256×64 = 16K
    - fc_out: 64×1 = 64
    - Total: ~0.47M
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
                             
        self.diff_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 8, embed_dim),
            nn.Sigmoid(),
        )
        
                                                  
        pair_dim = embed_dim * 3
        
                     
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )
    
    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                 
        raw_diff = torch.abs(f_a - f_b)
        
                   
        gate = self.diff_gate(raw_diff)
        gated_diff = raw_diff * gate
        
                       
        prod = f_a * f_b
        
                 
        pair = torch.cat([gated_diff, prod, raw_diff], dim=-1)
        
                   
        logits = self.mlp(pair).squeeze(-1)
        
        return logits, {
            "diff": raw_diff,
            "gated_diff": gated_diff,
            "prod": prod,
            "gate": gate,
        }


class SiameseHead(nn.Module):
    """Enhanced Siamese similarity head for FA pair discrimination.

    Enhancements:
    1) Multi-scale difference processing with learnable weights.
    2) Per-dimension difference weighting to emphasize discriminative features.
    3) Cross-feature interaction via bilinear-like attention.
    4) SE-style channel attention on concatenated pair features.
    5) Deeper MLP with residual connection for better gradient flow.
    """

    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int = 512, 
        diff_hidden_dim: int | None = None,
        num_diff_scales: int = 3,             
        use_bilinear: bool = True,             
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_bilinear = use_bilinear

        if diff_hidden_dim is None:
            diff_hidden_dim = embed_dim

                                  
        self.diff_scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, diff_hidden_dim // num_diff_scales),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_diff_scales)
        ])
        multi_scale_dim = (diff_hidden_dim // num_diff_scales) * num_diff_scales
        
                                                                
                                
        self.diff_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid(),
        )
        
                               
        if use_bilinear:
            bilinear_dim = embed_dim // 4
            self.bilinear_proj_a = nn.Linear(embed_dim, bilinear_dim)
            self.bilinear_proj_b = nn.Linear(embed_dim, bilinear_dim)
            self.bilinear_out = nn.Linear(bilinear_dim, bilinear_dim)
        else:
            bilinear_dim = 0

                                                                   
                                                                            
                                                                           
                                                              
                                                                      
                                                                        
                                                                         
                                                                        
        pair_dim = embed_dim + multi_scale_dim + (bilinear_dim if use_bilinear else embed_dim)

                                       
        se_hidden = max(pair_dim // 8, 16)
        self.se_net = nn.Sequential(
            nn.Linear(pair_dim, se_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(se_hidden, pair_dim),
            nn.Sigmoid(),
        )

                                          
        self.fc1 = nn.Linear(pair_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc_out = nn.Linear(hidden_dim // 4, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim // 2)
        
                                                       
        if hidden_dim != hidden_dim // 2:
            self.res_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        else:
            self.res_proj = nn.Identity()

    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                   
        raw_diff = torch.abs(f_a - f_b)
        
                                                               
        diff_weights = self.diff_weight(raw_diff)
        weighted_diff = raw_diff * diff_weights
        
                    
        diff_ms_list = [scale(raw_diff) for scale in self.diff_scales]
        diff_ms = torch.cat(diff_ms_list, dim=-1)
        
                     
        prod = f_a * f_b
        
                      
        if self.use_bilinear:
            proj_a = self.bilinear_proj_a(f_a)
            proj_b = self.bilinear_proj_b(f_b)
            bilinear = self.bilinear_out(proj_a * proj_b)
            pair = torch.cat([weighted_diff, diff_ms, bilinear], dim=-1)
        else:
            pair = torch.cat([weighted_diff, diff_ms, prod], dim=-1)

                                       
        se = self.se_net(pair)
        pair = pair * se

                                         
        x = F.relu(self.fc1(pair))
        x = self.dropout(x)
        
        x_res = self.res_proj(x)
        x = F.relu(self.fc2(x))
        x = self.layer_norm(x + x_res)                       
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        logits = self.fc_out(x)
        
        return logits.squeeze(-1), {
            "diff": raw_diff, 
            "weighted_diff": weighted_diff,
            "diff_ms": diff_ms, 
            "prod": prod,
            "diff_weights": diff_weights,
        }

