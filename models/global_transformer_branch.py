"""Global transformer branch using DINOv3 backbones."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from safetensors.torch import load_file as safe_load_file

try:
    import timm
except ImportError as exc:                                                   
    raise ImportError("timm is required for the transformer branch") from exc

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from transformers import AutoModel
except Exception:            
    AutoModel = None


@dataclass
class GlobalTransformerConfig:
                                                               
    backbone_type: str = "dino_vit"
                                               
    model_name: str = "vit_base_patch16_224.dinov3.lvd142m"
                                                      
    pretrained_path: Optional[str] = None
                                                     
    timm_pretrained: bool = True
                                                      
    hf_model_name: Optional[str] = None
                                                                    
    hub_repo: str = r"./dinov3"
                                                         
    hub_model_name: str = "dinov3_vits16"
    feature_dim: int = 512
    embed_dim: int = 256
    freeze: bool = True
    preconv_channels: int = 3
    drop_path_rate: float = 0.0
                                               
    use_preconv: bool = False


class GrayscaleToRGB(nn.Module):
    """Convert grayscale (1-channel) image to RGB (3-channel) for ViT input.

    Uses a learnable 3×3 Conv2d instead of channel expand() so the model can
    learn per-output-channel spatial filters rather than being restricted to
    per-channel affine transforms of the same grayscale signal.  Initialized
    as a near-identity (center pixel = 1, all other weights = 0, bias = 0) so
    the starting behavior is equivalent to simple channel replication.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == out_channels:
                                         
            self.proj: Optional[nn.Conv2d] = None
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
                                                                                        
                                              
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            center = self.proj.kernel_size[0] // 2               
                                                                                 
                                    
            self.proj.weight.data[:, :, center, center] = 1.0 / in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return x
        return self.proj(x)


class PreConvEmbedding(nn.Module):
    """Shallow conv stem placed before ViT to stabilize handwriting inputs.
    
    Deprecated: Use GrayscaleToRGB instead for simpler grayscale to RGB conversion.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class GlobalTransformerBranch(nn.Module):
    """ViT branch that captures holistic style cues."""

    def __init__(self, cfg: GlobalTransformerConfig, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
                              
        if cfg.use_preconv:
            self.preconv = PreConvEmbedding(in_channels, cfg.preconv_channels)
        else:
            self.preconv = GrayscaleToRGB(in_channels, cfg.preconv_channels)

                                                   
        self.transformer = None

        _load_errors: List[str] = []            

        if cfg.backbone_type == "dino_vit":
                                                                 
            if cfg.pretrained_path is not None:
                try:
                    print(
                        f"[GlobalTransformerBranch] Loading DINOv3 backbone via torch.hub: "
                        f"repo='{cfg.hub_repo}', model='{cfg.hub_model_name}'"
                    )
                    try:
                        self.transformer = torch.hub.load(
                            repo_or_dir=cfg.hub_repo,
                            model=cfg.hub_model_name,
                            source="local",
                            weights=str(cfg.pretrained_path),
                        )
                    except Exception:
                        self.transformer = torch.hub.load(
                            repo_or_dir=cfg.hub_repo,
                            model=cfg.hub_model_name,
                            source="local",
                        )
                except Exception as exc:
                    _load_errors.append(f"torch.hub: {exc}")
                    print(f"[GlobalTransformerBranch] torch.hub DINOv3 load failed: {exc}")

                                                                
            if self.transformer is None and cfg.hf_model_name is not None and AutoModel is not None:
                try:
                    print(
                        f"[GlobalTransformerBranch] Loading DINOv3 backbone via HuggingFace: '\n"
                        f"model={cfg.hf_model_name}'"
                    )
                    self.transformer = AutoModel.from_pretrained(cfg.hf_model_name)
                except Exception as exc:
                    _load_errors.append(f"HuggingFace: {exc}")
                    print(f"[GlobalTransformerBranch] HuggingFace DINOv3 load failed: {exc}")

                                                      
        if self.transformer is None:
            model_name = cfg.model_name
            base_name = model_name.split(".")[0]

                                                          
                                              
            if cfg.pretrained_path is not None and str(cfg.pretrained_path).endswith(".safetensors"):
                print(
                    f"[GlobalTransformerBranch] Building timm model '{model_name}' and "
                    f"loading weights from local safetensors: {cfg.pretrained_path}"
                )
                try:
                    self.transformer = _create_timm_model(
                        model_name,
                        pretrained=False,
                        num_classes=0,
                        drop_path_rate=cfg.drop_path_rate,
                    )
                except RuntimeError as exc:
                    if "Invalid pretrained tag" not in str(exc):
                        raise
                    print(
                        f"[GlobalTransformerBranch] '{model_name}' not supported by current timm install; "
                        f"falling back to '{base_name}' without a pretrained tag."
                    )
                    self.transformer = _create_timm_model(
                        base_name,
                        pretrained=False,
                        num_classes=0,
                        drop_path_rate=cfg.drop_path_rate,
                    )

                state = safe_load_file(str(cfg.pretrained_path), device="cpu")
                missing, unexpected = self.transformer.load_state_dict(state, strict=False)
                if missing:
                    print(f"[GlobalTransformerBranch] Missing keys when loading safetensors: {missing}")
                if unexpected:
                    print(f"[GlobalTransformerBranch] Unexpected keys when loading safetensors: {unexpected}")
            else:
                pretrained = cfg.timm_pretrained and cfg.pretrained_path is None
                try:
                    self.transformer = _create_timm_model(
                        model_name,
                        pretrained=pretrained,
                        num_classes=0,
                        drop_path_rate=cfg.drop_path_rate,
                    )
                except RuntimeError as exc:
                    if "Invalid pretrained tag" not in str(exc):
                        raise
                    print(
                        f"[GlobalTransformerBranch] '{model_name}' not supported by current timm install; "
                        f"falling back to '{base_name}' without a pretrained tag."
                    )
                    self.transformer = _create_timm_model(
                        base_name,
                        pretrained=False,
                        num_classes=0,
                        drop_path_rate=cfg.drop_path_rate,
                    )

                                                 
            if cfg.pretrained_path and cfg.backbone_type == "dino_vit" and str(cfg.pretrained_path).endswith(".pth"):
                state = torch.load(cfg.pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]

                new_state = {}
                for k, v in state.items():
                    if k.startswith("storage_tokens") or k.startswith("mask_token"):
                        continue
                    if k.startswith("rope_embed"):
                        continue
                    if "bias_mask" in k:
                        continue
                    if ".ls1." in k or ".ls2." in k:
                        continue
                    new_state[k] = v

                missing, unexpected = self.transformer.load_state_dict(new_state, strict=False)
                if missing:
                    print(
                        f"[GlobalTransformerBranch] Missing keys after DINOv3 remap: {missing}"
                    )
                if unexpected:
                    print(
                        f"[GlobalTransformerBranch] Unexpected keys after DINOv3 remap: {unexpected}"
                    )

                                              
        if self.transformer is None:
            error_details = "; ".join(_load_errors) if _load_errors else "Unknown error"
            raise RuntimeError(
                f"[GlobalTransformerBranch] Failed to load transformer backbone. "
                f"Tried methods: {error_details}. "
                f"Please check your config: backbone_type='{cfg.backbone_type}', "
                f"model_name='{cfg.model_name}', pretrained_path='{cfg.pretrained_path}'"
            )

        num_features = int(getattr(self.transformer, "num_features", cfg.feature_dim))
        self.embed = nn.Linear(num_features, cfg.feature_dim)
        self.embed_act = nn.GELU()
        self.proj = nn.Linear(cfg.feature_dim, cfg.embed_dim)

                                                                                    
                                                                                     
        if cfg.preconv_channels == 3:
            mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1)
        else:
            mean = torch.full((1, cfg.preconv_channels, 1, 1), 0.5, dtype=torch.float32)
            std = torch.full((1, cfg.preconv_channels, 1, 1), 0.5, dtype=torch.float32)
        self.register_buffer("_img_mean", mean, persistent=False)
        self.register_buffer("_img_std", std, persistent=False)
        if cfg.freeze:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def set_freeze_ratio(self, unfreeze_blocks: int = 0) -> None:
        """Unfreeze only the last *unfreeze_blocks* transformer layers."""
        for p in self.transformer.parameters():
            p.requires_grad = False
        if unfreeze_blocks <= 0:
            return
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None:
            return
        for block in blocks[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preconv(x)
        x = (x - self._img_mean) / self._img_std
        feat = self.transformer(x)
        feat = self.embed(feat)
        feat = self.embed_act(feat)
        feat = self.proj(feat)
                                                                                    
        return feat


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _create_timm_model(
    model_name: str,
    *,
    pretrained: bool,
    num_classes: int,
    drop_path_rate: float,
):
    kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
    }
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    try:
        return timm.create_model(model_name, **kwargs)
    except TypeError as exc:
        if "drop_path_rate" in kwargs and "drop_path_rate" in str(exc):
            print(
                f"[GlobalTransformerBranch] Model '{model_name}' does not accept drop_path_rate; "
                "retrying without stochastic depth."
            )
            kwargs.pop("drop_path_rate")
            return timm.create_model(model_name, **kwargs)
        raise
