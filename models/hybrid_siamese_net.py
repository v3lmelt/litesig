"""Hybrid Siamese network that wraps the hybrid backbone and similarity head."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from .hybrid_backbone import HybridBackbone, HybridBackboneConfig
from .siamese_head import SiameseHead, LightSiameseHead



@dataclass
class HybridSiameseConfig:
    backbone: HybridBackboneConfig = field(default_factory=HybridBackboneConfig)
    embed_dim: int = 512
    use_light_head: bool = False                    
    light_head_hidden_dim: int = 256                         
    light_head_dropout: float = 0.1                            
    use_forg_head: bool = False                                                                  
    use_writer_cls: bool = False                                                               
    writer_cls_num_classes: int = 0                                                           


class HybridSiameseNet(nn.Module):
    """Optimized Hybrid Siamese Network.
    
    Improvements:
    - Merged forward: concatenate both inputs and run backbone once
    - Reduces redundant computation for shared-weight siamese branches
    - Auto-detects BatchNorm layers and switches to separate forward to avoid
      batch statistics pollution between paired samples
    """
    
    def __init__(self, cfg: HybridSiameseConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = HybridBackbone(cfg.backbone)
        
                        
        embed_dim = cfg.backbone.fusion_embed_dim
        if cfg.use_light_head:
            self.head = LightSiameseHead(
                embed_dim=embed_dim,
                hidden_dim=cfg.light_head_hidden_dim,
                dropout=cfg.light_head_dropout,
            )
            print(f"[HybridSiameseNet] Using LightSiameseHead (hidden_dim={cfg.light_head_hidden_dim})")
        else:
            self.head = SiameseHead(embed_dim)
            print("[HybridSiameseNet] Using SiameseHead (full)")
        
                                                                             
        self.forg_head: Optional[nn.Module] = None
        if cfg.use_forg_head:
            _forg_hidden = max(embed_dim // 4, 64)
            self.forg_head = nn.Sequential(
                nn.Linear(embed_dim, _forg_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(_forg_hidden, 1),
            )
            print("[HybridSiameseNet] Forgery detection head enabled (aux task, HTCSigNet style)")

                                                                                      
        self.cls_head: Optional[nn.Module] = None
        if cfg.use_writer_cls and cfg.writer_cls_num_classes > 0:
            _cls_hidden = max(embed_dim // 2, 128)
            self.cls_head = nn.Sequential(
                nn.Linear(embed_dim, _cls_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(_cls_hidden, cfg.writer_cls_num_classes),
            )
            print(f"[HybridSiameseNet] Writer classification head enabled ({cfg.writer_cls_num_classes} classes)")

                                       
        self._contains_batchnorm = self._detect_batchnorm()
        if self._contains_batchnorm:
            print("[HybridSiameseNet] BatchNorm detected in backbone, using separate forward to avoid statistics pollution")
    
    def _detect_batchnorm(self) -> bool:
        """检测 backbone 是否包含 BatchNorm 层。
        
        如果存在 BatchNorm，merged forward 会导致两个样本共享 batch 统计量，
        导致训练/推理不一致。此时应使用 forward_separate。
        """
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False

    def forward_single(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a single input (used during inference with single image)."""
        return self.backbone(x)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor, skip_head: bool = False) -> Dict[str, object]:
        """Optimized forward: merge inputs, single backbone pass, then split.
        
        This reduces computation by ~40% for transformer backbones compared to
        running the backbone twice on separate inputs.
        
        Note: If backbone contains BatchNorm layers, this method automatically
        switches to forward_separate to avoid batch statistics pollution.
        
        Args:
            xa: First input tensor [B, C, H, W]
            xb: Second input tensor [B, C, H, W]
            skip_head: If True, skip Siamese head computation (useful when using
                      distance-based loss and head is frozen)
        """
                                                        
                                                            
        if self._contains_batchnorm:
            return self._forward_separate_impl(xa, xb, skip_head)
        
        batch_size = xa.size(0)
        
                                                  
        x_merged = torch.cat([xa, xb], dim=0)                 
        
                                      
        out_merged = self.backbone(x_merged)
        
                                              
        out_a = {}
        out_b = {}
        for key, value in out_merged.items():
            if isinstance(value, torch.Tensor):
                out_a[key] = value[:batch_size]
                out_b[key] = value[batch_size:]
            else:
                                                  
                out_a[key] = value
                out_b[key] = value
        
        fused_a = out_a["fused"]
        fused_b = out_b["fused"]

        result = {
            "fused_a": fused_a,
            "fused_b": fused_b,
            "branches_a": out_a,
            "branches_b": out_b,
        }
        
                                                              
        if self.forg_head is not None:
            result["forg_logits_a"] = self.forg_head(fused_a).squeeze(-1)       
            result["forg_logits_b"] = self.forg_head(fused_b).squeeze(-1)       

                                                                  
        if self.cls_head is not None:
            result["cls_logits_a"] = self.cls_head(fused_a)                    
            result["cls_logits_b"] = self.cls_head(fused_b)                    

                                                                                    
        if not skip_head:
            logits, aux = self.head(fused_a, fused_b)
            result["logits"] = logits
            result["aux"] = aux
        
        return result

    def _forward_separate_impl(self, xa: torch.Tensor, xb: torch.Tensor, skip_head: bool = False) -> Dict[str, object]:
        """Separate forward implementation that avoids BatchNorm statistics pollution.
        
        Each input is processed independently through the backbone, ensuring
        BatchNorm layers compute statistics separately for each sample.
        
        Args:
            xa: First input tensor [B, C, H, W]
            xb: Second input tensor [B, C, H, W]
            skip_head: If True, skip Siamese head computation
        """
        out_a = self.forward_single(xa)
        out_b = self.forward_single(xb)
        
        fused_a = out_a["fused"]
        fused_b = out_b["fused"]

        result = {
            "fused_a": fused_a,
            "fused_b": fused_b,
            "branches_a": out_a,
            "branches_b": out_b,
        }
        
                                                              
        if self.forg_head is not None:
            result["forg_logits_a"] = self.forg_head(fused_a).squeeze(-1)       
            result["forg_logits_b"] = self.forg_head(fused_b).squeeze(-1)       

                                                                  
        if self.cls_head is not None:
            result["cls_logits_a"] = self.cls_head(fused_a)                    
            result["cls_logits_b"] = self.cls_head(fused_b)                    

        if not skip_head:
            logits, aux = self.head(fused_a, fused_b)
            result["logits"] = logits
            result["aux"] = aux

        return result

    def forward_separate(self, xa: torch.Tensor, xb: torch.Tensor, skip_head: bool = False) -> Dict[str, object]:
        """Original separate forward (kept for compatibility/debugging).
        
        Use this when inputs have different batch sizes or for debugging.
        
        Args:
            xa: First input tensor [B, C, H, W]
            xb: Second input tensor [B, C, H, W]
            skip_head: If True, skip Siamese head computation
        """
        return self._forward_separate_impl(xa, xb, skip_head)

    def stage1_freeze(self) -> bool:
        """Freeze only the ViT transformer weights for stage-1 training.

        Projection layers (embed, embed_act, proj, preconv) remain trainable so
        the new head can adapt even while the backbone is frozen.

        Returns True if the transformer was found and frozen, False otherwise.
        """
        transformer = getattr(
            getattr(self.backbone, "global_branch", None), "transformer", None
        )
        if transformer is None:
            return False
        for p in transformer.parameters():
            p.requires_grad = False
        return True

    def stage2_unfreeze_backbone(self) -> bool:
        """Fully unfreeze ViT transformer weights for stage-2 fine-tuning.

        Returns True if the transformer was found and unfrozen, False otherwise.
        """
        transformer = getattr(
            getattr(self.backbone, "global_branch", None), "transformer", None
        )
        if transformer is None:
            return False
        for p in transformer.parameters():
            p.requires_grad = True
        return True

    def stage2_partial_unfreeze(self, blocks: int = 2) -> None:
        """Partially unfreeze the last *blocks* transformer blocks for stage-2 fine-tuning.

        This is an alternative to stage2_unfreeze_backbone(): instead of fully
        unfreezing the transformer it re-freezes all transformer params first and
        then re-enables only the last *blocks* blocks (via set_freeze_ratio).
        Projection layers (embed, embed_act, proj, preconv) are unaffected — they
        remain in whatever state they were in before this call.

        Use stage2_unfreeze_backbone() if you want to unfreeze the entire transformer.
        """
        if self.backbone.global_branch:
            self.backbone.global_branch.set_freeze_ratio(blocks)
