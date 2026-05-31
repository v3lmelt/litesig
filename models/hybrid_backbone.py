"""Hybrid backbone combining the LiteSig local/global branches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from .conv_branch import LocalConvBranch, ConstantChannelLocalConvBranch
from .fusion_head import FusionHead
from .global_transformer_branch import GlobalTransformerBranch, GlobalTransformerConfig


@dataclass
class HybridBackboneConfig:
    use_local_branch: bool = True
    use_global_branch: bool = True
                                                                               
                                                                           
    use_vmamba_branch: bool = False
    use_projection_histogram_branch: bool = False
    use_frequency_aware_branch: bool = False
    use_dinov3_convnext_branch: bool = False
    use_custom_convnext_branch: bool = False
    use_custom_swin_branch: bool = False
    use_multiscale_cnn_branch: bool = False
    use_modified_crossvit_branch: bool = False
    use_resnet_fpn_branch: bool = False
    local_feature_dim: int = 256
    local_num_blocks: int = 3
                                   
    local_use_constant_channel: bool = True
    local_constant_channels: int = 64
                                        
    local_use_sda: bool = True
    local_sda_groups: int = 64
    local_use_sda_last_block: bool = False
                                             
    local_use_channel_spatial_attn: bool = False
    local_csam_ratio: int = 2
                                          
    local_use_lrsa: bool = False
    local_lrsa_num_heads: int = 2
    local_lrsa_q_pooled_size: int = 2
    local_lrsa_pooled_sizes: Optional[Sequence[int]] = None
                                            
    local_use_msc_fusion: bool = False
    local_msc_low_layer_idx: int = 0
    local_msc_num_heads: int = 4
    local_msc_max_spatial_size: Optional[int] = None
                                      
    local_use_ssp: bool = True
    local_ssp_blocks: int = 1
                                         
    local_use_fsc: bool = True
    local_input_height: int = 256                         
    local_input_width: int = 512                          
    global_feature_dim: int = 256
    fusion_embed_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_identity: bool = False                                                  
    in_channels: int = 1
    transformer: GlobalTransformerConfig = field(default_factory=GlobalTransformerConfig)


class HybridBackbone(nn.Module):
    """Multi-branch feature extractor with configurable components."""

    def __init__(self, cfg: HybridBackboneConfig):
        super().__init__()
        self.cfg = cfg
        branches = []
        input_dims = []
        if cfg.use_local_branch:
            if cfg.local_use_constant_channel:
                                                                                              
                self.local_branch = ConstantChannelLocalConvBranch(
                    in_channels=cfg.in_channels,
                    constant_channels=cfg.local_constant_channels,
                    num_blocks=cfg.local_num_blocks,
                    feature_dim=cfg.local_feature_dim,
                    use_wavelet=True,
                    use_ssp=cfg.local_use_ssp,
                    ssp_blocks=cfg.local_ssp_blocks,
                    use_fsc=cfg.local_use_fsc,
                    input_height=cfg.local_input_height,
                    input_width=cfg.local_input_width,
                    use_sda=cfg.local_use_sda,
                    sda_groups=cfg.local_sda_groups,
                    use_channel_spatial_attn=cfg.local_use_channel_spatial_attn,
                    csam_ratio=cfg.local_csam_ratio,
                    use_lrsa=cfg.local_use_lrsa,
                    lrsa_num_heads=cfg.local_lrsa_num_heads,
                    lrsa_q_pooled_size=cfg.local_lrsa_q_pooled_size,
                    lrsa_pooled_sizes=cfg.local_lrsa_pooled_sizes,
                )
                print(
                    f"[HybridBackbone] Using ConstantChannelLocalConvBranch: "
                    f"channels={cfg.local_constant_channels}, blocks={cfg.local_num_blocks}, "
                    f"feature_dim={cfg.local_feature_dim}"
                )
            else:
                                 
                self.local_branch = LocalConvBranch(
                    in_channels=cfg.in_channels,
                    num_blocks=cfg.local_num_blocks,
                    feature_dim=cfg.local_feature_dim,
                    use_sda=cfg.local_use_sda,
                    sda_groups=cfg.local_sda_groups,
                    use_sda_last_block=cfg.local_use_sda_last_block,
                    use_channel_spatial_attn=cfg.local_use_channel_spatial_attn,
                    csam_ratio=cfg.local_csam_ratio,
                    use_lrsa=cfg.local_use_lrsa,
                    lrsa_num_heads=cfg.local_lrsa_num_heads,
                    lrsa_q_pooled_size=cfg.local_lrsa_q_pooled_size,
                    lrsa_pooled_sizes=cfg.local_lrsa_pooled_sizes,
                    use_msc_fusion=cfg.local_use_msc_fusion,
                    msc_low_layer_idx=cfg.local_msc_low_layer_idx,
                    msc_num_heads=cfg.local_msc_num_heads,
                    msc_max_spatial_size=cfg.local_msc_max_spatial_size,
                    use_ssp=cfg.local_use_ssp,
                    ssp_blocks=cfg.local_ssp_blocks,
                    use_fsc=cfg.local_use_fsc,
                    input_height=cfg.local_input_height,
                    input_width=cfg.local_input_width,
                )
            branches.append("local")
            input_dims.append(cfg.local_feature_dim)
        else:
            self.local_branch = None

        if cfg.use_global_branch:
            gt_cfg = cfg.transformer
            gt_cfg.feature_dim = cfg.global_feature_dim * 2
            gt_cfg.embed_dim = cfg.global_feature_dim
            self.global_branch = GlobalTransformerBranch(gt_cfg, in_channels=cfg.in_channels)
            branches.append("global")
            input_dims.append(cfg.global_feature_dim)
        else:
            self.global_branch = None

        removed_branches = (
            (cfg.use_vmamba_branch, "VMamba"),
            (cfg.use_projection_histogram_branch, "projection histogram"),
            (cfg.use_frequency_aware_branch, "frequency-aware"),
            (cfg.use_dinov3_convnext_branch, "DINOv3 ConvNeXt"),
            (cfg.use_custom_convnext_branch, "custom ConvNeXt"),
            (cfg.use_custom_swin_branch, "custom Swin"),
            (cfg.use_multiscale_cnn_branch, "multi-scale CNN"),
            (cfg.use_modified_crossvit_branch, "modified CrossViT"),
            (cfg.use_resnet_fpn_branch, "ResNet-FPN"),
        )
        enabled_removed = [name for enabled, name in removed_branches if enabled]
        if enabled_removed:
            raise ValueError(
                "This minimal release only supports the local/global LiteSig branches "
                "used by train/train.py. Removed branch(es) requested: "
                + ", ".join(enabled_removed)
            )

        if not input_dims:
            raise ValueError("At least one branch must be enabled")
        self.fusion = FusionHead(input_dims, embed_dim=cfg.fusion_embed_dim, dropout=cfg.fusion_dropout, identity=cfg.fusion_identity)
        self.enabled_branches = branches


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if self.local_branch:
            local_feat = self.local_branch(x)
            outputs["local"] = local_feat
        if self.global_branch:
            global_feat = self.global_branch(x)
            outputs["global"] = global_feat
        fused = self.fusion(list(outputs.values()))
        outputs["fused"] = fused
        return outputs
