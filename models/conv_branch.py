"""Local CNN branch centered on SPDConv + SCConv modules."""
from __future__ import annotations

from typing import List, Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .scconv import SCConv, SCConvConfig
from .spd_conv import SPDConv, SPDConvConfig
from components.spatial_attention import DynamicSpatialAttention
from components.convutr import ConvUtr
from components.stroke_directional_attention import StrokeDirectionalAttention
from components.channel_spatial_attn import CSAM
from components.low_resolution_self_attn import LRSA
from components.wavelet import WTFD
from components.stroke_scale_perception import StrokeScalePerceptionConv
from components.multilevel_fusion import MSC
from components.fourier_stroke_calibration import FourierStrokeCalibration


class MainWithSSP(nn.Module):
    """Parallel residual structure combining the main branch with SSP."""
    def __init__(
        self,
        main_seq: nn.Sequential,
        ssp_seq: nn.Sequential,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.main = main_seq
        self.ssp = ssp_seq

        if in_channels != out_channels:
            self.ssp_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.ssp_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        ssp_in = self.ssp_proj(x)
        ssp_out = self.ssp(ssp_in)
        if ssp_out.shape[2:] != main_out.shape[2:]:
            ssp_out = F.adaptive_avg_pool2d(ssp_out, output_size=main_out.shape[2:])
        return main_out + ssp_out


class LRSAWrapper(nn.Module):
    """Wrapper for LRSA to make it compatible with nn.Sequential.
    
    LRSA requires H and W as additional arguments, but nn.Sequential
    only passes the tensor. This wrapper extracts H and W from the input tensor.
    """
    def __init__(self, lrsa_module: nn.Module):
        super().__init__()
        self.lrsa = lrsa_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                                      
        _, _, H, W = x.shape
        return self.lrsa(x, H, W)


class ResidualWrapper(nn.Module):
    """Residual wrapper for dimension-preserving modules.
    
    Implements: y = x + F(x)
    
    适用于输入输出维度完全相同的模块（相同 channels, H, W）。
    用于包装 ConvUtr + SCConv 组合，提供残差连接以改善梯度流。
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class ParallelSCConvConvUtr(nn.Module):
    """Parallel Local/Global unit: SCConv (local) || ConvUtr (global) + concat fusion.
    
    设计说明:
    1. Local 分支 (SCConv): 专注于局部纹理/笔画细节
    2. Global 分支 (ConvUtr): 专注于大感受野/全局结构
    3. 两条路径并行处理，互不干扰
    4. 融合: concat([local, global]) -> Conv1x1 -> BN -> ReLU
    5. 残差固定开启: out = x + gamma * fused
    
    这种设计解决了串行堆叠中"局部特征被全局算子平滑"或"全局特征被局部算子打碎"的问题。
    """
    def __init__(
        self,
        channels: int,
        scconv_kernel: int = 3,
        convutr_kernel: int = 9,
        convutr_depth: int = 1,
        gamma_init: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
                                   
        self.local = SCConv(
            SCConvConfig(
                in_channels=channels,
                out_channels=channels,
                kernel_size=scconv_kernel,
                stride=1,
                use_bn=use_bn,
            )
        )
        
                                         
        self.global_branch = ConvUtr(
            ch_in=channels,
            ch_out=channels,
            depth=convutr_depth,
            kernel=convutr_kernel,
        )
        
                                                    
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
                                
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
              
        y_local = self.local(x)
        y_global = self.global_branch(x)
        
            
        y_cat = torch.cat([y_local, y_global], dim=1)
        y_fused = self.fuse(y_cat)
        
                     
        return x + self.gamma * y_fused


class FSCParallelBranch(nn.Module):
    """FSC as an independent parallel branch."""
    def __init__(
        self,
        main_branch: nn.Module,
        in_channels: int,
        out_channels: int,
        input_height: int,
        input_width: int,
        init_gamma: float = 0.1,
    ):
        super().__init__()
        self.main = main_branch
        self.expected_height = input_height
        self.expected_width = input_width

        if in_channels != out_channels:
            self.fsc_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.fsc_proj = nn.Identity()

        self.fsc = FourierStrokeCalibration(
            input_chs=out_channels,
            output_chs=out_channels,
            num_rows=input_height,
            num_cols=input_width,
            stride=1,
            init='he',
        )

        self.gamma = nn.Parameter(torch.tensor(init_gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        fsc_in = self.fsc_proj(x)
        if fsc_in.shape[2:] != main_out.shape[2:]:
            fsc_in = F.adaptive_avg_pool2d(fsc_in, output_size=main_out.shape[2:])
        fsc_out = self.fsc(fsc_in)
        return main_out + self.gamma * fsc_out


class LocalConvBranch(nn.Module):
    """HTCSigNet-inspired local branch that keeps stroke details.
    
    Architecture (per block):
    - SPDConv -> (WTFD preprocessing) -> (LRSA optional) -> Parallel(SCConv || ConvUtr) -> (SSP optional)
    
    Key design:
    - Parallel Local/Global: SCConv (local textures) || ConvUtr (global receptive field)
    - concat + 1x1 fusion with learnable gamma residual (fixed enabled)
    - MSC cross-layer feature fusion (optional)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 1,
        scconv_kernel_sizes: Sequence[int] | None = None,
        feature_dim: int = 64,
        use_group_norm: bool = False,                        
                                       
        use_ssp: bool = True,
        ssp_blocks: int = 1,
                                            
        use_convutr: bool = True,
        convutr_depth: int = 1,
        convutr_kernel_sizes: Sequence[int] | None = None,
                                        
        use_wavelet: bool = True,
                                   
        use_residual: bool = True,
                      
        use_msc_fusion: bool = False,
        msc_low_layer_idx: int = 0,                                            
        msc_num_heads: int = 4,
        msc_k1: int = 2,
        msc_k2: int = 3,
        msc_max_spatial_size: Optional[int] = None,
                                            
        use_sda: bool = True,
        sda_groups: int = 64,
                                                        
        use_sda_last_block: bool = False,
                                                 
        use_channel_spatial_attn: bool = False,
        csam_ratio: int = 2,
                                              
        use_lrsa: bool = False,
        lrsa_num_heads: int = 2,
        lrsa_q_pooled_size: int = 2,
        lrsa_pooled_sizes: Optional[List[int]] = None,
                                          
        use_fsc: bool = True,
        input_height: int = 256,
        input_width: int = 512,
    ) -> None:
        super().__init__()
        self.use_msc_fusion = use_msc_fusion
        self.msc_low_layer_idx = msc_low_layer_idx
        self.num_blocks = num_blocks
        self.use_sda = use_sda
        self.use_sda_last_block = use_sda_last_block

        if use_sda and use_sda_last_block:
            import warnings
            warnings.warn(
                "Both use_sda and use_sda_last_block are enabled. "
                "This may cause redundant Stroke-Directional Attention stacking. "
                "Consider using only one of them for efficiency.",
                UserWarning,
            )
        
        scconv_kernel_sizes = scconv_kernel_sizes or [3] * num_blocks
        convutr_kernel_sizes = convutr_kernel_sizes or [9] * num_blocks                              

                                                      
                                                   
                                     
        if use_wavelet:
                                           
                                                             
                                           
                                                     
                                            
            self.wavelet_preprocess = WTFD(in_channels, in_channels, mode="fusion")
        else:
            self.wavelet_preprocess = None

                                               
        self.blocks = nn.ModuleList()
        self.block_out_channels: List[int] = []
        
        c_in = in_channels
        last_block_idx = num_blocks - 1
        for idx in range(num_blocks):
            c_out = base_channels * (2**idx)
            self.block_out_channels.append(c_out)
            
            spd = SPDConv(
                SPDConvConfig(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=3,
                    reduction=2 if idx > 0 else 1,
                )
            )
                                                                      
                                          
            main_block: List[nn.Module] = []
            main_block.append(spd)


                                                                       
                                                        
            if use_lrsa and idx == last_block_idx:
                lrsa_module = LRSA(
                    dim=c_out,
                    num_heads=lrsa_num_heads,
                    q_pooled_size=lrsa_q_pooled_size,
                    pooled_sizes=lrsa_pooled_sizes or [11, 8, 6, 4]
                )
                                           
                main_block.append(LRSAWrapper(lrsa_module))

                                                                 
                                          
                                                    
            main_block.append(
                ParallelSCConvConvUtr(
                    channels=c_out,
                    scconv_kernel=scconv_kernel_sizes[idx],
                    convutr_kernel=convutr_kernel_sizes[idx],
                    convutr_depth=convutr_depth,
                    gamma_init=0.1,
                    use_bn=True,
                )
            )

            if use_ssp:
                ssp_layers: List[nn.Module] = []
                for _ in range(ssp_blocks):
                    ssp_layers.append(StrokeScalePerceptionConv(c_out))
                current_block = MainWithSSP(
                    main_seq=nn.Sequential(*main_block),
                    ssp_seq=nn.Sequential(*ssp_layers),
                    in_channels=c_in,
                    out_channels=c_out,
                )
            else:
                current_block = nn.Sequential(*main_block)

            if use_fsc:
                cumulative_reduction = 1 if idx == 0 else 2 ** idx
                feat_h = max(1, input_height // cumulative_reduction)
                feat_w = max(1, input_width // cumulative_reduction)

                current_block = FSCParallelBranch(
                    main_branch=current_block,
                    in_channels=c_in,
                    out_channels=c_out,
                    input_height=feat_h,
                    input_width=feat_w,
                    init_gamma=0.1
                )
            
            self.blocks.append(current_block)

            c_in = c_out

                 
        self._spatial_channels = c_in

        self.sda = (
            StrokeDirectionalAttention(inp=c_in, oup=c_in, groups=sda_groups)
            if use_sda
            else None
        )

        self.sda_last_block = (
            StrokeDirectionalAttention(inp=c_in, oup=c_in, groups=sda_groups)
            if use_sda_last_block
            else None
        )

                                                 
        self.csam = (
            CSAM(in_channels=c_in, ratio=csam_ratio)
            if use_channel_spatial_attn
            else None
        )

                      
        if use_msc_fusion and num_blocks >= 2:
                     
            low_channels = self.block_out_channels[msc_low_layer_idx]
                                 
            high_channels = self.block_out_channels[-1]
            
                                     
            if low_channels != high_channels:
                self.low_to_high_proj = nn.Sequential(
                    nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(high_channels),
                )
            else:
                self.low_to_high_proj = nn.Identity()
            
                                                  
                                                              
            max_spatial_size = msc_max_spatial_size
            if max_spatial_size is None:
                max_spatial_size = min(64, int(max(input_height, input_width)))
            self.msc = MSC(
                dim=high_channels,
                num_heads=msc_num_heads,
                kernel=[3, 5, 7],
                s=[1, 1, 1],
                pad=[1, 2, 3],
                qkv_bias=True,
                attn_drop_ratio=0.1,
                proj_drop_ratio=0.1,
                k1=msc_k1,
                k2=msc_k2,
                max_spatial_size=max_spatial_size,
            )
            
                                                             
                                                 
                                               
            self.msc_gamma = nn.Parameter(torch.tensor(0.1))
        else:
            self.msc = None
            self.low_to_high_proj = None
            self.msc_gamma = None

                   
        norm = nn.GroupNorm(8, c_in) if use_group_norm else nn.BatchNorm2d(c_in)
        self.final_norm = norm
        self.final_act = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                                       
                                           
        if self.wavelet_preprocess is not None:
            x = self.wavelet_preprocess(x)
        
                             
        intermediate_feats: List[torch.Tensor] = []
        feat = x
        
        last_block_idx = len(self.blocks) - 1
        for idx, block in enumerate(self.blocks):
            feat = block(feat)

            if idx == last_block_idx and self.sda_last_block is not None:
                feat = self.sda_last_block(feat)

            intermediate_feats.append(feat)
        
                    
        if self.msc is not None and self.low_to_high_proj is not None:
                         
            feat_low = intermediate_feats[self.msc_low_layer_idx]                            
            feat_high = feat                                               
            
                            
            feat_low_proj = self.low_to_high_proj(feat_low)                             
            
                                                             
                                                 
                                               
                                          
            original_high_size = feat_high.shape[2:]
            if feat_low_proj.shape[2:] != feat_high.shape[2:]:
                feat_high_upsampled = F.interpolate(
                    feat_high,
                    size=feat_low_proj.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feat_high_upsampled = feat_high
            
                                                        
                                    
            msc_out = self.msc(feat_high_upsampled, feat_low_proj)                             
            
                                                   
            if msc_out.shape[2:] != original_high_size:
                msc_out = F.adaptive_avg_pool2d(msc_out, output_size=original_high_size)
            
                           
            feat = feat_high + self.msc_gamma * msc_out

        if self.sda is not None:
            feat = self.sda(feat)

                                                 
        if self.csam is not None:
            feat = self.csam(feat)

                  
        feat = self.final_norm(feat)
        feat = self.final_act(feat)
        
        self._last_feature_map = feat
        feat = self.pool(feat).flatten(1)
        feat = self.fc(feat)
                                                                                    
        return feat
    @property
    def last_feature_map(self) -> torch.Tensor | None:
        return getattr(self, "_last_feature_map", None)

    @property
    def spatial_channels(self) -> int:
        return self._spatial_channels


class ParallelSCConvDilated(nn.Module):
    """Parallel local/dilated unit for constant-channel architecture.

    Structure:
    - Local branch: SCConv (dilation=1, captures fine stroke textures)
    - Dilated branch: depthwise-separable conv with dilation (expands receptive field)
    - Fusion: cat([local, dilated], dim=1) -> Conv1x1 -> BN -> ReLU
    - Output: x + gamma * fused (learnable residual)

    This replaces ParallelSCConvConvUtr for the constant-channel design.
    Instead of using ConvUtr (large decomposed kernels), we use dilated
    depthwise-separable convolutions which are much lighter and naturally
    scale receptive field with block index (dilation=2^idx).
    """

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        gamma_init: float = 0.1,
    ):
        super().__init__()
                                                          
        self.local = SCConv(
            SCConvConfig(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                use_bn=True,
            )
        )

                                                                
        self.dilated = nn.Sequential(
                                                                       
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
                                               
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

                                                        
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

                                   
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = self.local(x)
        y_dilated = self.dilated(x)
        y_cat = torch.cat([y_local, y_dilated], dim=1)
        y_fused = self.fuse(y_cat)
        return x + self.gamma * y_fused


class ConstantChannelLocalConvBranch(nn.Module):
    """Constant-channel local branch that preserves spatial resolution.

    Key design differences from LocalConvBranch:
    1. ALL blocks use the same channel width (no exponential growth)
    2. NO spatial downsampling within blocks (preserves stroke detail)
    3. Receptive field grows via dilated convolutions (dilation=2^idx)
    4. FSC and SSP are applied once at the end
    5. Pre-Norm residual blocks for stable multi-block stacking

    Architecture:
        Input -> [WTFD] -> Stem(SPDConv, no downsample) ->
        Block0(dilation=1) -> Block1(dilation=2) -> ... -> BlockN(dilation=2^N) ->
        [SSP once] -> [FSC once] -> [SDA/CSAM] ->
        BN -> ReLU -> AdaptiveAvgPool2d -> Linear -> Output

    This design ensures parameters grow LINEARLY with num_blocks,
    making it safe to stack many blocks without overfitting.
    """

    def __init__(
        self,
        in_channels: int = 1,
        constant_channels: int = 64,
        num_blocks: int = 3,
        feature_dim: int = 256,
        gamma_init: float = 0.1,
                               
        use_wavelet: bool = True,
                                                     
        use_ssp: bool = True,
        ssp_blocks: int = 1,
                                                        
        use_fsc: bool = True,
        input_height: int = 224,
        input_width: int = 224,
                                            
        use_sda: bool = True,
        sda_groups: int = 64,
                                                 
        use_channel_spatial_attn: bool = False,
        csam_ratio: int = 2,
                                                                     
        use_lrsa: bool = False,
        lrsa_num_heads: int = 2,
        lrsa_q_pooled_size: int = 2,
        lrsa_pooled_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        C = constant_channels

                                                     
        if use_wavelet:
            self.wavelet_preprocess = WTFD(in_channels, in_channels, mode="fusion")
        else:
            self.wavelet_preprocess = None

                                                                                    
        self.stem = SPDConv(
            SPDConvConfig(
                in_channels=in_channels,
                out_channels=C,
                kernel_size=3,
                reduction=1,                   
            )
        )


                                                                             
        self.blocks = nn.ModuleList()
        self.block_norms = nn.ModuleList()                           

        for idx in range(num_blocks):
            dilation = 2 ** idx                   
            self.block_norms.append(nn.BatchNorm2d(C))
            self.blocks.append(
                ParallelSCConvDilated(
                    channels=C,
                    dilation=dilation,
                    gamma_init=gamma_init,
                )
            )

                                                                     
        if use_lrsa:
            lrsa_module = LRSA(
                dim=C,
                num_heads=lrsa_num_heads,
                q_pooled_size=lrsa_q_pooled_size,
                pooled_sizes=lrsa_pooled_sizes or [11, 8, 6, 4],
            )
            self.lrsa = LRSAWrapper(lrsa_module)
        else:
            self.lrsa = None

        if use_ssp:
            ssp_layers: List[nn.Module] = []
            for _ in range(ssp_blocks):
                ssp_layers.append(StrokeScalePerceptionConv(C))
            self.ssp = nn.Sequential(*ssp_layers)
            self.ssp_gamma = nn.Parameter(torch.tensor(gamma_init))
        else:
            self.ssp = None
            self.ssp_gamma = None

        if use_fsc:
            self.fsc = FourierStrokeCalibration(
                input_chs=C,
                output_chs=C,
                num_rows=input_height,
                num_cols=input_width,
                stride=1,
                init='he',
            )
            self.fsc_norm = nn.BatchNorm2d(C)
            self.fsc_gamma = nn.Parameter(torch.tensor(gamma_init))
        else:
            self.fsc = None
            self.fsc_norm = None
            self.fsc_gamma = None

        self.sda = (
            StrokeDirectionalAttention(inp=C, oup=C, groups=sda_groups)
            if use_sda
            else None
        )

        self.csam = (
            CSAM(in_channels=C, ratio=csam_ratio)
            if use_channel_spatial_attn
            else None
        )

                                            
        self._spatial_channels = C
        self.final_norm = nn.BatchNorm2d(C)
        self.final_act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                               
        if self.wavelet_preprocess is not None:
            x = self.wavelet_preprocess(x)

                                                         
        feat = self.stem(x)


                                                                
        for idx in range(self.num_blocks):
                                                                              
            normed = self.block_norms[idx](feat)
            feat = self.blocks[idx](normed)

                                           
        if self.lrsa is not None:
            feat = self.lrsa(feat)

        if self.ssp is not None:
            ssp_out = self.ssp(feat)
            feat = feat + self.ssp_gamma * ssp_out

        if self.fsc is not None:
            fsc_out = self.fsc(feat)
            feat = feat + self.fsc_gamma * fsc_out

        if self.sda is not None:
            feat = self.sda(feat)
        if self.csam is not None:
            feat = self.csam(feat)

                      
        feat = self.final_norm(feat)
        feat = self.final_act(feat)

        self._last_feature_map = feat
        feat = self.pool(feat).flatten(1)
        feat = self.fc(feat)
        return feat

    @property
    def last_feature_map(self) -> torch.Tensor | None:
        return getattr(self, "_last_feature_map", None)

    @property
    def spatial_channels(self) -> int:
        return self._spatial_channels


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
