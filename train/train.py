from __future__ import annotations

import argparse
import json
import math
import sys
import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, cast
from datetime import datetime
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.global_transformer_branch import GlobalTransformerConfig
from models.hybrid_backbone import HybridBackboneConfig
from models.hybrid_siamese_net import HybridSiameseConfig, HybridSiameseNet
from utils.dataset_chhd_writer_disjoint import ChhdWriterDisjointDataset
from utils.dataset_writer_disjoint import WriterDisjointDataset
from utils.dataset_writer_disjoint_npz import NpzWriterDisjointDataset
from utils.loss import (
    LossConfig,
    contrastive_loss_signature_verification,
    contrastive_loss_with_hard_mining,
    focal_contrastive_loss,
    bce_distance_loss,
    smooth_contrastive_loss,
)


LOCAL_NUM_BLOCKS = 3
LOCAL_USE_CONSTANT_CHANNEL = True
LOCAL_CONSTANT_CHANNELS = 64
LOCAL_SDA_GROUPS = 64


@dataclass
class ErrorSampleInfo:

    rank: int
    score: float
    gt_label: int
    pred_label: int
    thr_used: float
    path_a: Optional[str] = None
    path_b: Optional[str] = None

    @property
    def gt_text(self) -> str:
        return "Same" if self.gt_label == 1 else "Different"

    @property
    def pred_text(self) -> str:
        return "Same" if self.pred_label == 1 else "Different"


def create_annotated_error_image(
    sample: ErrorSampleInfo,
    target_height: int = 224,
) -> Optional[torch.Tensor]:

    if not sample.path_a or not sample.path_b:
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
        import torchvision.transforms.functional as TF


        img_a = Image.open(sample.path_a).convert("RGB")
        img_b = Image.open(sample.path_b).convert("RGB")


        def resize_to_height(img: Image.Image, h: int) -> Image.Image:
            w_orig, h_orig = img.size
            if h_orig == h:
                return img
            ratio = h / h_orig
            new_w = int(w_orig * ratio)
            return img.resize((new_w, h), Image.LANCZOS)

        img_a = resize_to_height(img_a, target_height)
        img_b = resize_to_height(img_b, target_height)


        w_a, h_a = img_a.size
        w_b, h_b = img_b.size
        gap = 4
        combined_w = w_a + gap + w_b
        combined_h = h_a

        combined = Image.new("RGB", (combined_w, combined_h), color=(255, 255, 255))
        combined.paste(img_a, (0, 0))
        combined.paste(img_b, (w_a + gap, 0))


        annotation_height = 60
        final_h = combined_h + annotation_height
        final_img = Image.new("RGB", (combined_w, final_h), color=(255, 255, 255))
        final_img.paste(combined, (0, annotation_height))


        draw = ImageDraw.Draw(final_img)


        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()


        lines = [
            f"GT: {sample.gt_text}  |  Pred: {sample.pred_text}",
            f"score: {sample.score:.4f}  |  thr: {sample.thr_used:.4f}",
        ]


        y_offset = 5
        for line in lines:
            draw.text((5, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += 22


        tensor = TF.to_tensor(final_img)
        return tensor

    except Exception as e:

        return None


def print_model_summary(model: torch.nn.Module, model_name: str = "Model") -> None:

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\n" + "=" * 80)
    print(f"  {model_name} Summary")
    print("=" * 80)
    print(f"  Total parameters:     {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params / 1e6:.2f}M)")
    print(f"  Trainable ratio:      {100 * trainable_params / total_params:.2f}%")
    print("=" * 80)
    print("\n  Model Structure:")
    print("-" * 80)
    print(model)
    print("-" * 80 + "\n")


def compute_binary_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    if labels.numel() == 0:
        return float("nan")
    unique = labels.unique(sorted=True)
    if unique.numel() < 2:
        return float("nan")
    sorted_scores, indices = torch.sort(scores)
    sorted_labels = labels[indices]
    pos_count = sorted_labels.sum().item()
    neg_count = sorted_labels.numel() - pos_count
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[sorted_labels == 1].sum().item()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def _sweep_frr_far(
    labels: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    positives = labels == 1
    negatives = labels == 0
    pos_total = positives.sum().item()
    neg_total = negatives.sum().item()
    if pos_total == 0 or neg_total == 0:
        return (
            torch.full((1,), float("nan")),
            torch.full((1,), float("nan")),
            torch.full((1,), float("nan")),
        )
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1 - sorted_labels, dim=0)
    fnr = (pos_total - tps) / pos_total
    fpr = fps / neg_total
    return fnr, fpr, sorted_scores


def compute_error_rates(
    labels: torch.Tensor, scores: torch.Tensor
) -> Tuple[float, float, float]:
    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    if labels.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    fnr, fpr, thresholds = _sweep_frr_far(labels, scores)
    if torch.isnan(fnr).all() or torch.isnan(fpr).all():
        return float("nan"), float("nan"), float("nan")
    diff = torch.abs(fnr - fpr)
    idx = int(torch.argmin(diff).item())
    frr = float(fnr[idx].item())
    far = float(fpr[idx].item())
    thr = float(thresholds[idx].item())
    return frr, far, thr


def compute_eer(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    if labels.numel() == 0:
        return float("nan")
    fnr, fpr, _ = _sweep_frr_far(labels, scores)
    if torch.isnan(fnr).all() or torch.isnan(fpr).all():
        return float("nan")
    diff = torch.abs(fnr - fpr)
    idx = int(torch.argmin(diff).item())
    eer = float((fnr[idx] + fpr[idx]) * 0.5)
    return eer


def compute_best_accuracy(labels: torch.Tensor, scores: torch.Tensor) -> tuple[float, float]:

    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    total = labels.numel()
    pos_total = float((labels == 1).sum().item())
    neg_total = float((labels == 0).sum().item())
    if total == 0 or pos_total == 0 or neg_total == 0:
        return float("nan"), float("nan")
    fnr, fpr, thresholds = _sweep_frr_far(labels, scores)
    if torch.isnan(fnr).all() or torch.isnan(fpr).all():
        return float("nan"), float("nan")
    error = fnr * pos_total + fpr * neg_total
    acc = 1.0 - (error / total)
    idx = int(torch.argmax(acc).item())
    return float(acc[idx].item()), float(thresholds[idx].item())


def format_lrs(optimizer: torch.optim.Optimizer) -> str:
    lrs = [group["lr"] for group in optimizer.param_groups]
    return ",".join(f"{lr:.2e}" for lr in lrs)


def _amp_autocast(device_type: str, enabled: bool, dtype_str: str = "bf16"):

    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(device_type=device_type, enabled=enabled, dtype=dtype)
        except TypeError:

            return amp_mod.autocast(enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _create_grad_scaler(device_type: str, enabled: bool):

    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler(device_type, enabled=enabled)
        except TypeError:

            return amp_mod.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class WarmupCosineScheduler:


    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0





        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(0)):
            param_group['lr'] = lr

    def get_lr(self, step: Optional[int] = None) -> list:
        if step is None:
            step = self.current_step

        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / self.warmup_steps
            else:
                progress = min(
                    1.0,
                    (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps),
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                min_lr = base_lr * self.min_lr_ratio
                lr = min_lr + (base_lr - min_lr) * cosine_decay
            lrs.append(lr)
        return lrs

    def step(self):


        self.current_step += 1
        lrs = self.get_lr(self.current_step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def state_dict(self) -> dict:
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr_ratio': self.min_lr_ratio,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict):
        self.current_step = state_dict['current_step']
        self.total_steps = state_dict['total_steps']
        self.warmup_steps = state_dict['warmup_steps']
        self.min_lr_ratio = state_dict['min_lr_ratio']
        self.base_lrs = state_dict['base_lrs']


class RandomThickness:

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        from PIL import ImageFilter

        if torch.rand(1).item() < 0.5:

            return img.filter(ImageFilter.MaxFilter(3))
        else:

            return img.filter(ImageFilter.MinFilter(3))


class ElasticTransform:


    def __init__(self, alpha: float = 15.0, sigma: float = 4.0, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return img

        C, H, W = img.shape


        dx = torch.randn(1, H, W) * self.alpha
        dy = torch.randn(1, H, W) * self.alpha


        kernel_size = int(self.sigma * 4) | 1
        if kernel_size > 3:
            pad = kernel_size // 2
            dx = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(dx.unsqueeze(0), (pad, pad, pad, pad), mode='replicate'),
                kernel_size, stride=1
            ).squeeze(0)
            dy = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(dy.unsqueeze(0), (pad, pad, pad, pad), mode='replicate'),
                kernel_size, stride=1
            ).squeeze(0)


        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )


        grid_x = grid_x + dx.squeeze(0) / (W / 2)
        grid_y = grid_y + dy.squeeze(0) / (H / 2)

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        img_transformed = torch.nn.functional.grid_sample(
            img.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)

        return img_transformed


class ForegroundRandomErasing:


    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.1),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        fg_threshold: float = 0.05,
        max_attempts: int = 10,
    ) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.fg_threshold = fg_threshold
        self.max_attempts = max_attempts

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        if not torch.is_tensor(img) or img.dim() != 3:
            return img

        _, height, width = img.shape
        if height <= 1 or width <= 1:
            return img

        if img.size(0) == 1:
            fg_mask = img[0] > self.fg_threshold
        else:
            fg_mask = (img > self.fg_threshold).any(dim=0)

        if fg_mask.sum().item() == 0:
            return img

        ys, xs = torch.nonzero(fg_mask, as_tuple=True)
        area = height * width

        for _ in range(self.max_attempts):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).item())
            erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if erase_h < 1 or erase_w < 1:
                continue
            if erase_h > height or erase_w > width:
                continue

            idx = torch.randint(0, ys.numel(), (1,)).item()
            cy = int(ys[idx].item())
            cx = int(xs[idx].item())

            top = cy - erase_h // 2
            left = cx - erase_w // 2
            top = max(0, min(top, height - erase_h))
            left = max(0, min(left, width - erase_w))

            img[:, top:top + erase_h, left:left + erase_w] = float(self.value)
            return img

        return img


def build_train_transforms(
    height: int,
    width: int,
    use_thickness_aug: bool = False,
    thickness_aug_p: float = 0.3,
    use_elastic: bool = False,
    elastic_alpha: float = 15.0,
    elastic_sigma: float = 4.0,
    elastic_p: float = 0.3,
):


    pre_tensor = [T.Resize((height, width))]


    if use_thickness_aug:
        pre_tensor.append(RandomThickness(p=thickness_aug_p))

    pre_tensor.append(
        T.RandomApply(
            [
                T.RandomAffine(
                    degrees=3.0,
                    translate=(0.1, 0.1),
                    scale=(0.95, 1.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0,
                )
            ],
            p=0.6,
        )
    )

    pre_tensor.append(T.ToTensor())

    post_tensor = []


    if use_elastic:
        post_tensor.append(
            ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, p=elastic_p)
        )

    post_tensor.append(
        ForegroundRandomErasing(
            p=0.5,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0.0,
            fg_threshold=0.05,
            max_attempts=10,
        )
    )

    return T.Compose(pre_tensor + post_tensor)


def build_val_transforms(height: int, width: int):
    return T.Compose(
        [
            T.Resize((height, width)),
            T.ToTensor(),
        ]
    )


def build_datasets(
    data_root: str,
    train_transform,
    val_transform,
    writer_split_ratio: float,
    split_seed: int,
    train_positive_pairs: int | None,
    train_negative_pairs: int | None,
    val_positive_pairs: int | None,
    val_negative_pairs: int | None,
    pair_sampling_seed: int | None,
    balance_pairs: bool,
    oversample: bool = False,
    partitions: Optional[Dict[str, List[str]]] = None,
    match_by_sig: bool = False,
    npz_path: Optional[str] = None,
    input_channels: int = 1,
    dataset_format: str = "standard",
    forged_subsets: Optional[Sequence[str]] = None,
) -> Tuple[
    WriterDisjointDataset | ChhdWriterDisjointDataset | NpzWriterDisjointDataset,
    WriterDisjointDataset | ChhdWriterDisjointDataset | NpzWriterDisjointDataset,
]:

    if npz_path is not None:
        train_ds = NpzWriterDisjointDataset(
            npz_path,
            split="train",
            transform=train_transform,
            writer_split_ratio=writer_split_ratio,
            split_seed=split_seed,
            partitions=partitions,
            positive_pairs_per_writer=train_positive_pairs,
            negative_pairs_per_writer=train_negative_pairs,
            pair_sampling_seed=pair_sampling_seed,
            balance_pairs=balance_pairs,
            oversample=oversample,
            input_channels=input_channels,
        )
        val_ds = NpzWriterDisjointDataset(
            npz_path,
            split="val",
            transform=val_transform,
            partitions=train_ds.partitions,
            positive_pairs_per_writer=val_positive_pairs,
            negative_pairs_per_writer=val_negative_pairs,
            pair_sampling_seed=pair_sampling_seed,
            balance_pairs=balance_pairs,
            input_channels=input_channels,
        )
        return train_ds, val_ds

    if dataset_format == "chhd":
        train_ds = ChhdWriterDisjointDataset(
            data_root,
            split="train",
            transform=train_transform,
            writer_split_ratio=writer_split_ratio,
            split_seed=split_seed,
            partitions=partitions,
            positive_pairs_per_writer=train_positive_pairs,
            negative_pairs_per_writer=train_negative_pairs,
            pair_sampling_seed=pair_sampling_seed,
            balance_pairs=balance_pairs,
            oversample=oversample,
            input_channels=input_channels,
        )
        val_ds = ChhdWriterDisjointDataset(
            data_root,
            split="val",
            transform=val_transform,
            partitions=train_ds.partitions,
            positive_pairs_per_writer=val_positive_pairs,
            negative_pairs_per_writer=val_negative_pairs,
            pair_sampling_seed=pair_sampling_seed,
            balance_pairs=balance_pairs,
            input_channels=input_channels,
        )
        return train_ds, val_ds

    train_ds = WriterDisjointDataset(
        data_root,
        split="train",
        transform=train_transform,
        writer_split_ratio=writer_split_ratio,
        split_seed=split_seed,
        partitions=partitions,
        positive_pairs_per_writer=train_positive_pairs,
        negative_pairs_per_writer=train_negative_pairs,
        pair_sampling_seed=pair_sampling_seed,
        balance_pairs=balance_pairs,
        oversample=oversample,
        match_by_sig=match_by_sig,
        input_channels=input_channels,
        forged_subsets=forged_subsets,
    )
    val_ds = WriterDisjointDataset(
        data_root,
        split="val",
        transform=val_transform,
        partitions=train_ds.partitions,
        positive_pairs_per_writer=val_positive_pairs,
        negative_pairs_per_writer=val_negative_pairs,
        pair_sampling_seed=pair_sampling_seed,
        balance_pairs=balance_pairs,
        match_by_sig=match_by_sig,
        input_channels=input_channels,
        forged_subsets=forged_subsets,
    )
    return train_ds, val_ds


def build_dataloaders(
    train_ds: WriterDisjointDataset | ChhdWriterDisjointDataset | NpzWriterDisjointDataset,
    val_ds: WriterDisjointDataset | ChhdWriterDisjointDataset | NpzWriterDisjointDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def summarize_train_pairs(
    train_ds: WriterDisjointDataset | ChhdWriterDisjointDataset | NpzWriterDisjointDataset,
) -> None:
    total = len(train_ds)
    if total == 0:
        print("[DINO] Train dataset is empty.")
        return
    pos = sum(1 for s in train_ds.samples if s.forgery_label == 0)
    neg = total - pos
    pos_ratio = pos / total
    neg_ratio = neg / total
    print(
        f"[DINO] Train pairs: total={total} pos={pos} ({pos_ratio:.3f}) neg={neg} ({neg_ratio:.3f})"
    )


def refresh_train_pairs_for_epoch(
    train_ds: WriterDisjointDataset | ChhdWriterDisjointDataset | NpzWriterDisjointDataset,
    epoch: int,
    base_pair_seed: int,
) -> int:

    epoch_seed = int(base_pair_seed) + int(epoch) - 1
    setattr(train_ds, "_pair_seed", epoch_seed)
    target_writers = train_ds.partitions["train"]
    train_ds.samples = train_ds._build_pairs(target_writers)
    return epoch_seed


def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    amp_dtype: str = "bf16",
    max_grad_norm: float = 1.0,
    scheduler: Optional[WarmupCosineScheduler] = None,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 1,
    use_logits_loss: bool = False,
    use_forg_head: bool = False,
    forg_lambda: float = 0.5,
    use_writer_cls: bool = False,
    writer_cls_lambda: float = 0.5,
) -> Tuple[float, float, float, float]:
    model.train()



    try:
        backbone = getattr(model, "backbone", None)
        global_branch = getattr(backbone, "global_branch", None)
        transformer = getattr(global_branch, "transformer", None)
        if transformer is not None:

            params = list(transformer.parameters())
            if params and (not params[0].requires_grad):
                transformer.eval()
    except Exception:

        pass
    running = 0.0
    running_forg = 0.0
    running_cls = 0.0

    train_preds_list = []
    train_labels_list = []

    pbar = tqdm(loader, desc="DINO Train", leave=False)
    amp_enabled = use_amp and device.type == "cuda"
    steps_per_epoch = len(loader)

    for batch_idx, batch in enumerate(pbar):
        xa = batch["img_a"].to(device, non_blocking=True)
        xb = batch["img_b"].to(device, non_blocking=True)
        same = batch["same_writer"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_autocast(device.type, amp_enabled, amp_dtype):

            if use_logits_loss:

                out = model(xa, xb, skip_head=False)
                logits = out["logits"]
                total = loss_fn(logits, same)

                scores = logits.detach()
            else:

                out = model(xa, xb, skip_head=True)
                feat_a = out["fused_a"]
                feat_b = out["fused_b"]
                total = loss_fn(feat_a, feat_b, same)

                dist = torch.norm(feat_a - feat_b, dim=1)
                scores = -dist



            if use_forg_head:
                forg_label_a = batch["forg_label_a"].to(device, non_blocking=True)
                forg_label_b = batch["forg_label_b"].to(device, non_blocking=True)
                forg_logits_a = out["forg_logits_a"]
                forg_logits_b = out["forg_logits_b"]

                forg_logits = torch.cat([forg_logits_a, forg_logits_b], dim=0)
                forg_labels = torch.cat([forg_label_a, forg_label_b], dim=0)
                forg_loss = torch.nn.functional.binary_cross_entropy_with_logits(forg_logits, forg_labels)
                total = total + forg_lambda * forg_loss
                running_forg += forg_loss.item()


            if use_writer_cls:
                wid = batch["writer_id"].to(device, non_blocking=True)
                cls_logits = torch.cat([out["cls_logits_a"], out["cls_logits_b"]], dim=0)
                cls_targets = torch.cat([wid, wid], dim=0)
                cls_loss = torch.nn.functional.cross_entropy(cls_logits, cls_targets)
                total = total + writer_cls_lambda * cls_loss
                running_cls += cls_loss.item()

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()


        train_preds_list.append(scores.detach().cpu())
        train_labels_list.append(same.detach().cpu())

        if scheduler is not None:
            if writer is not None:
                global_step = (epoch - 1) * steps_per_epoch + batch_idx
                writer.add_scalar("lr/step", optimizer.param_groups[0]["lr"], global_step)
            scheduler.step()

        running += total.item()
        avg_loss = running / (pbar.n or 1)
        avg_forg_loss = running_forg / (pbar.n or 1)
        avg_cls_loss = running_cls / (pbar.n or 1)
        postfix = dict(loss=f"{avg_loss:.4f}", forg=f"{avg_forg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if use_writer_cls:
            postfix["cls"] = f"{avg_cls_loss:.4f}"
        pbar.set_postfix(**postfix)




    train_acc = float("nan")
    train_acc_at_05 = float("nan")
    if train_preds_list:
        train_preds_cat = torch.cat(train_preds_list)
        train_labels_cat = torch.cat(train_labels_list)
        if use_logits_loss:
            probs = torch.sigmoid(train_preds_cat)
            preds = (probs >= 0.5).float()
            train_acc_at_05 = float((preds == train_labels_cat).float().mean().item())
            train_acc, _ = compute_best_accuracy(train_labels_cat, train_preds_cat)
        else:
            train_acc, _ = compute_best_accuracy(train_labels_cat, train_preds_cat)


    avg_forg_loss = running_forg / len(loader)

    return running / len(loader), train_acc, train_acc_at_05, avg_forg_loss


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp: bool, amp_dtype: str = "bf16", error_topk: int = 20, use_logits_loss: bool = False):

    model.eval()
    total = 0.0


    logits_list = []
    labels_list = []


    sample_info = []

    pbar = tqdm(loader, desc="DINO Val", leave=False)
    amp_enabled = use_amp and device.type == "cuda"

    for batch_idx, batch in enumerate(pbar):
        xa = batch["img_a"].to(device, non_blocking=True)
        xb = batch["img_b"].to(device, non_blocking=True)
        same = batch["same_writer"].to(device, non_blocking=True)

        with _amp_autocast(device.type, amp_enabled, amp_dtype):
            if use_logits_loss:

                out = model(xa, xb, skip_head=False)
                logits = out["logits"]
                scores = logits
                total_loss = loss_fn(logits, same)
            else:

                out = model(xa, xb, skip_head=True)
                feat_a = out["fused_a"]
                feat_b = out["fused_b"]
                dist = torch.norm(feat_a - feat_b, dim=1)
                scores = -dist
                total_loss = loss_fn(feat_a, feat_b, same)

        total += total_loss.item()


        scores_cpu = scores.detach().cpu()
        labels_cpu = same.detach().cpu()
        logits_list.append(scores_cpu)
        labels_list.append(labels_cpu)


        batch_size = xa.size(0)
        has_paths = "path_a" in batch and "path_b" in batch
        for i in range(batch_size):
            info = {
                "score": float(scores_cpu[i].item()),
                "label": int(labels_cpu[i].item()),
                "path_a": batch["path_a"][i] if has_paths else None,
                "path_b": batch["path_b"][i] if has_paths else None,
            }
            sample_info.append(info)

        avg_loss = total / (pbar.n or 1)
        pbar.set_postfix(loss=avg_loss)

    mean_loss = total / len(loader)
    fa_count = 0
    fr_count = 0
    fa_items: List[ErrorSampleInfo] = []
    fr_items: List[ErrorSampleInfo] = []

    if logits_list:
        logits_cat = torch.cat(logits_list)
        labels_cat = torch.cat(labels_list)
        auc = compute_binary_auc(labels_cat, logits_cat)

        frr, far, thr = compute_error_rates(labels_cat, logits_cat)
        if not math.isnan(thr):
            preds = (logits_cat >= thr).float()
            acc = float((preds == labels_cat.float()).float().mean().item())
            thr_used = thr
        else:



            thr_used = float(logits_cat.median().item())
            preds = (logits_cat >= thr_used).float()
            acc = float((preds == labels_cat.float()).float().mean().item())


        acc_best, thr_best = compute_best_accuracy(labels_cat, logits_cat)



        acc_at_05 = float("nan")
        far_at_05 = float("nan")
        frr_at_05 = float("nan")
        if use_logits_loss:

            preds_at_05 = (logits_cat >= 0).float()
            acc_at_05 = float((preds_at_05 == labels_cat).float().mean().item())


            pos_mask = (labels_cat == 1)
            neg_mask = (labels_cat == 0)
            pos_total = pos_mask.sum().item()
            neg_total = neg_mask.sum().item()
            if pos_total > 0:

                fr_at_05 = ((pos_mask) & (preds_at_05 == 0)).sum().item()
                frr_at_05 = float(fr_at_05 / pos_total)
            if neg_total > 0:

                fa_at_05 = ((neg_mask) & (preds_at_05 == 1)).sum().item()
                far_at_05 = float(fa_at_05 / neg_total)


            preds = preds_at_05


        fnr, fpr, thresholds_sweep = _sweep_frr_far(labels_cat, logits_cat)
        if not torch.isnan(fnr).all() and not torch.isnan(fpr).all():

            _, idx_best = torch.min(torch.abs(thresholds_sweep - thr_best), dim=0)
            idx_best = int(idx_best.item())
            frr_best = float(fnr[idx_best].item())
            far_best = float(fpr[idx_best].item())

            eer_at_best_acc = (frr_best + far_best) / 2.0
        else:
            frr_best = float("nan")
            far_best = float("nan")
            eer_at_best_acc = float("nan")

        eer = compute_eer(labels_cat, logits_cat)


        try:

            fa_candidates = [
                (info["score"], idx, info["path_a"], info["path_b"])
                for idx, info in enumerate(sample_info)
                if info["label"] == 0 and preds[idx] == 1
            ]
            fa_count = len(fa_candidates)


            fa_candidates.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, idx, pa, pb) in enumerate(fa_candidates[:error_topk], 1):
                fa_items.append(ErrorSampleInfo(
                    rank=rank,
                    score=score,
                    gt_label=0,
                    pred_label=1,
                    thr_used=thr_used,
                    path_a=pa,
                    path_b=pb,
                ))


            fr_candidates = [
                (info["score"], idx, info["path_a"], info["path_b"])
                for idx, info in enumerate(sample_info)
                if info["label"] == 1 and preds[idx] == 0
            ]
            fr_count = len(fr_candidates)


            fr_candidates.sort(key=lambda x: x[0])
            for rank, (score, idx, pa, pb) in enumerate(fr_candidates[:error_topk], 1):
                fr_items.append(ErrorSampleInfo(
                    rank=rank,
                    score=score,
                    gt_label=1,
                    pred_label=0,
                    thr_used=thr_used,
                    path_a=pa,
                    path_b=pb,
                ))

        except Exception as e:
            print(f"[DINO] Warning when computing top false-positives: {e}")
    else:
        auc = float("nan")
        acc = float("nan")
        acc_best = float("nan")
        acc_at_05 = float("nan")
        frr = float("nan")
        far = float("nan")
        thr = float("nan")
        thr_best = float("nan")
        eer = float("nan")
        frr_best = float("nan")
        far_best = float("nan")
        eer_at_best_acc = float("nan")
        far_at_05 = float("nan")
        frr_at_05 = float("nan")

    return mean_loss, auc, acc, acc_best, acc_at_05, frr, far, eer, thr, thr_best, frr_best, far_best, eer_at_best_acc, far_at_05, frr_at_05, fa_count, fr_count, fa_items, fr_items


def parse_args():
    parser = argparse.ArgumentParser(description="Single-stage training with DINO ViT backbone")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="standard",
        choices=["standard", "chhd"],
        help="Folder dataset format: standard=Genuine/Forged subdirs (nested Forged/* allowed), chhd=flat genuine-only writer dataset with cross-writer negatives.",
    )
    parser.add_argument(
        "--forged-subsets",
        nargs="+",
        default=None,
        help="Optional top-level subfolders under Forged/ to include. Useful for UTSig, e.g. --forged-subsets Skilled. "
             "Ignored when omitted.",
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default=None,
        help="Use a prebuilt NPZ dataset file. Overrides --data-root.",
    )
    parser.add_argument("--dino-ckpt", default=None, help="Path to DINO ViT or CrossViT weights")
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="dino_vit",
        choices=["dino_vit", "cross_vit", "timm_vit"],
        help="Backbone type: dino_vit (default), cross_vit (timm CrossViT), or timm_vit (generic timm ViT).",
    )
    parser.add_argument(
        "--backbone-model-name",
        type=str,
        default=None,
        help="timm model name for cross_vit or timm_vit backbone (e.g. vit_small_patch16_224.augreg_in21k_ft_in1k). Ignored for dino_vit.",
    )
    parser.add_argument(
        "--timm-pretrained",
        dest="timm_pretrained",
        action="store_true",
        default=None,
        help="Use timm-provided pretrained weights for timm_vit/cross_vit when no local checkpoint is loaded.",
    )
    parser.add_argument(
        "--no-timm-pretrained",
        dest="timm_pretrained",
        action="store_false",
        help="Do not use timm-provided pretrained weights for timm_vit/cross_vit.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="LR multiplier for DINO backbone (smaller for pretrained weights).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--writer-split-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--train-positive-pairs",
        type=int,
        default=None,
        help="Number of positive (genuine/genuine) pairs to draw per writer for training.",
    )
    parser.add_argument(
        "--train-negative-pairs",
        type=int,
        default=None,
        help="Number of negative pairs to draw per writer for training. Standard datasets use genuine/forged pairs; CHHD mode uses cross-writer genuine/genuine pairs.",
    )
    parser.add_argument(
        "--val-positive-pairs",
        type=int,
        default=None,
        help="Number of positive (genuine/genuine) pairs to draw per writer for validation. "
             "If not provided, defaults to --train-positive-pairs value.",
    )
    parser.add_argument(
        "--val-negative-pairs",
        type=int,
        default=None,
        help="Number of negative pairs to draw per writer for validation. "
             "Standard datasets use genuine/forged pairs; CHHD mode uses cross-writer genuine/genuine pairs. "
             "If not provided, defaults to --train-negative-pairs value.",
    )
    parser.add_argument(
        "--pair-sampling-seed",
        type=int,
        default=None,
        help="Seed for the random pair sampler when using per-writer pair caps.",
    )
    parser.set_defaults(resample_train_pairs_each_epoch=False)
    parser.add_argument(
        "--resample-train-pairs-each-epoch",
        dest="resample_train_pairs_each_epoch",
        action="store_true",
        help="Resample train pairs every epoch with deterministic seeds: epoch_seed = base_seed + (epoch - 1).",
    )
    parser.add_argument(
        "--no-resample-train-pairs-each-epoch",
        dest="resample_train_pairs_each_epoch",
        action="store_false",
        help="Disable per-epoch train pair resampling (use fixed pairs built at startup).",
    )
    parser.add_argument(
        "--no-balance-pairs",
        action="store_true",
        help="Disable automatic positive/negative balancing after pair generation.",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="If balancing pairs, use oversampling instead of downsampling.",
    )
    parser.add_argument(
        "--match-by-sig",
        action="store_true",
        help="Only pair samples with matching signature id (Hansig style). "
             "When enabled, positive pairs are genuine samples with same (writer, sig), "
             "and negative pairs are genuine vs forged with same (writer, sig).",
    )
    parser.add_argument(
        "--npz-info",
        action="store_true",
        help="Print NPZ dataset metadata and exit.",
    )
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        choices=[1, 3],
        help="Input image channels. Default: 1 for folder datasets, 3 for NPZ datasets.",
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable training data augmentation (use only Resize+ToTensor).",
    )

    parser.add_argument(
        "--use-elastic",
        action="store_true",
        help="Enable ElasticTransform augmentation (post-ToTensor).",
    )
    parser.add_argument(
        "--elastic-alpha",
        type=float,
        default=15.0,
        help="ElasticTransform displacement magnitude.",
    )
    parser.add_argument(
        "--elastic-sigma",
        type=float,
        default=4.0,
        help="ElasticTransform Gaussian smoothing sigma.",
    )
    parser.add_argument(
        "--elastic-p",
        type=float,
        default=0.3,
        help="ElasticTransform probability.",
    )
    parser.add_argument(
        "--use-thickness-aug",
        action="store_true",
        help="Enable RandomThickness augmentation (pre-ToTensor, PIL domain).",
    )
    parser.add_argument(
        "--thickness-aug-p",
        type=float,
        default=0.3,
        help="RandomThickness probability.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint", default="checkpoints/train.ckpt")
    parser.add_argument(
        "--tf-logs-root",
        type=str,
        default="/root/tf-logs",
        help="Root directory for TensorBoard logs",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (restores optimizer/scheduler/epoch).",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to a checkpoint to initialize model weights from (finetune mode: only loads model, resets optimizer/scheduler/epoch to start fresh).",
    )
    parser.add_argument(
        "--always-load-dino",
        action="store_true",
        help="When used with --init-from, still initialize the DINO backbone using --dino-ckpt (avoids timm-only fallback).",
    )

    parser.set_defaults(init_strict=True)
    parser.add_argument(
        "--no-init-strict",
        dest="init_strict",
        action="store_false",
        help="Use strict=False when loading --init-from weights (allows missing/unexpected keys).",
    )
    parser.add_argument(
        "--reinit-head",
        action="store_true",
        help="When used with --init-from, discard pretrained head (and forg_head) weights and reinitialize them randomly.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze DINO backbone (only train fusion layers).",
    )
    parser.add_argument(
        "--stage1-freeze-epochs",
        type=int,
        default=0,
        help="Two-stage training: freeze DINO backbone for the first N epochs, then unfreeze automatically. 0 = disabled.",
    )
    parser.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        action="store_true",
        help="Enable automatic mixed precision for faster training",
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable automatic mixed precision",
    )
    parser.set_defaults(mixed_precision=True)
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype: bf16 (better for Ampere GPUs) or fp16 (higher precision). (default: bf16)",
    )

    parser.add_argument("--use-local-branch", action="store_true", default=True)
    parser.add_argument("--no-local-branch", dest="use_local_branch", action="store_false")
    parser.add_argument("--use-global-branch", action="store_true", default=True)
    parser.add_argument("--no-global-branch", dest="use_global_branch", action="store_false")
    parser.add_argument(
        "--fusion-identity",
        action="store_true",
        help="Use identity fusion (concat + L2 norm only, no MLP). Paper mode.",
    )
    parser.add_argument(
        "--use-sda",
        dest="use_sda",
        action="store_true",
        default=True,
        help="Enable Stroke-Directional Attention (SDA) in the local branch (default: True).",
    )
    parser.add_argument(
        "--no-sda",
        dest="use_sda",
        action="store_false",
        help="Disable Stroke-Directional Attention (SDA).",
    )
    parser.add_argument(
        "--sda-last-block",
        dest="sda_last_block",
        action="store_true",
        help="Apply an additional SDA block right after the last LocalConvBranch block output.",
    )
    parser.add_argument(
        "--use-channel-spatial-attn",
        action="store_true",
        help="Enable the Channel-Spatial Attention Module (CSAM) in local branch.",
    )
    parser.add_argument(
        "--csam-ratio",
        type=int,
        default=2,
        help="Ratio parameter for CSAM (default: 2).",
    )
    parser.add_argument(
        "--use-lrsa",
        action="store_true",
        help="Enable Low-Resolution Self-Attention (LRSA) in local branch.",
    )
    parser.add_argument(
        "--lrsa-num-heads",
        type=int,
        default=2,
        help="Number of attention heads for LRSA (default: 2).",
    )
    parser.add_argument(
        "--lrsa-q-pooled-size",
        type=int,
        default=2,
        help="Q pooled size for LRSA (default: 2).",
    )

    parser.add_argument(
        "--use-msc-fusion",
        action="store_true",
        help="Enable MSC (Multi-Scale Cross) feature fusion in local branch.",
    )
    parser.add_argument(
        "--msc-low-layer-idx",
        type=int,
        default=0,
        help="Index of low-layer feature to use as Key/Value in MSC (default: 0, first block).",
    )
    parser.add_argument(
        "--msc-num-heads",
        type=int,
        default=4,
        help="Number of attention heads for MSC (default: 4).",
    )

    parser.add_argument(
        "--use-fsc",
        dest="use_fsc",
        action="store_true",
        default=True,
        help="Enable Fourier Stroke Calibration (FSC) in the local branch (default: True).",
    )
    parser.add_argument(
        "--no-fsc",
        dest="use_fsc",
        action="store_false",
        help="Disable Fourier Stroke Calibration (FSC).",
    )
    parser.add_argument(
        "--local-input-height",
        type=int,
        default=None,
        help="Input height for local branch (used to calculate FSC feature map sizes; defaults to --img-height).",
    )
    parser.add_argument(
        "--local-input-width",
        type=int,
        default=None,
        help="Input width for local branch (used to calculate FSC feature map sizes; defaults to --img-width).",
    )

    parser.add_argument(
        "--use-ssp",
        dest="use_ssp",
        action="store_true",
        default=True,
        help="Enable Stroke-Scale Perception (SSP) in the local branch (default: True).",
    )
    parser.add_argument(
        "--no-ssp",
        dest="use_ssp",
        action="store_false",
        help="Disable Stroke-Scale Perception (SSP).",
    )
    parser.add_argument(
        "--ssp-blocks",
        dest="ssp_blocks",
        metavar="SSP_BLOCKS",
        type=int,
        default=1,
        help="Number of SSP blocks in the local branch (default: 1).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=1.0,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Ratio of total steps for warmup (alternative to --warmup-epochs)",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Minimum LR ratio at end of cosine annealing",
    )
    parser.add_argument(
        "--use-epoch-scheduler",
        action="store_true",
        help="Use epoch-based scheduler instead of step-based",
    )
    parser.add_argument(
        "--exponential-lr-gamma",
        type=float,
        default=None,
        help="If set, use ExponentialLR with this gamma (paper: exponential decay each epoch).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin for contrastive loss",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")

    parser.add_argument(
        "--loss-type",
        type=str,
        default="contrastive",
        choices=["contrastive", "hard_mining", "focal", "bce_distance", "smooth", "bce_logits"],
        help="Loss function type: contrastive (baseline), hard_mining (Hard Negative Mining), focal (Focal Contrastive), bce_distance (BCE on distance, no saturation), smooth (smooth contrastive, no saturation), bce_logits (BCEWithLogitsLoss on head logits)",
    )
    parser.add_argument(
        "--bce-pos-weight",
        type=float,
        default=1.0,
        help="Positive class weight for BCEWithLogitsLoss (only used when --loss-type=bce_logits). Set >1 if positive samples are rare.",
    )
    parser.add_argument(
        "--hard-neg-ratio",
        type=float,
        default=0.5,
        help="Ratio of hardest negative samples to emphasize (0-1, for hard_mining loss)",
    )
    parser.add_argument(
        "--hard-pos-ratio",
        type=float,
        default=0.3,
        help="Ratio of hardest positive samples to emphasize (0-1, for hard_mining loss)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for focal loss (higher = more focus on hard samples)",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Early stopping patience (epochs without improvement). 0 = disabled.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0-0.2 recommended). Helps prevent overconfident predictions.",
    )
    parser.add_argument(
        "--dropout-rate",
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for regularization (0.0-0.5)",
    )
    parser.add_argument(
        "--head-weight-decay-mult",
        type=float,
        default=1.0,
        help="Multiplier for weight decay applied to model.head parameters (e.g., 2.0 means 2x).",
    )
    parser.add_argument(
        "--use-light-head",
        action="store_true",
        help="Use LightSiameseHead instead of the full SiameseHead.",
    )
    parser.add_argument(
        "--light-head-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for LightSiameseHead (default: 256).",
    )
    parser.add_argument(
        "--light-head-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for LightSiameseHead (default: 0.1).",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for this training session. If not provided, will prompt interactively.",
    )
    parser.add_argument(
        "--partitions-json",
        type=str,
        default=None,
        help="Path to a JSON file containing writer partitions for train/val splits. "
             "If provided, overrides --writer-split-ratio and --split-seed.",
    )
    parser.add_argument(
        "--error-topk",
        type=int,
        default=20,
        help="Number of top error samples (FA/FR) to log to TensorBoard per epoch. Default: 20.",
    )
    parser.add_argument(
        "--use-forg-head",
        action="store_true",
        help="Enable forgery detection auxiliary head (HTCSigNet style). "
             "Adds binary classification loss on each embedding to detect forged signatures.",
    )
    parser.add_argument(
        "--forg-lambda",
        type=float,
        default=0.5,
        help="Weight for forgery detection loss (λ in total = main_loss + λ*forg_loss). Default: 0.5.",
    )
    parser.add_argument(
        "--use-writer-cls",
        action="store_true",
        help="Enable writer classification auxiliary head (N-way softmax). "
             "Adds cross-entropy loss on each embedding to classify writer identity.",
    )
    parser.add_argument(
        "--writer-cls-lambda",
        type=float,
        default=0.5,
        help="Weight for writer classification loss (λ in total = main_loss + λ*cls_loss). Default: 0.5.",
    )
    return parser.parse_args()


def _sanitize_run_name(name: str) -> str:

    cleaned = name.strip()
    if not cleaned:
        return "default_run"

    cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")

    cleaned = re.sub(r'[\\/<>:"|?*]', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)

    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "default_run"


def build_optimizer_dino(
    model: torch.nn.Module,
    base_lr: float,
    backbone_lr_mult: float,
    weight_decay: float,
    head_weight_decay_mult: float = 1.0,
):
    params_backbone = []
    params_head = []
    params_others = []

    head_param_ids: set[int] = set()
    head = getattr(model, "head", None)
    if head is not None:
        for p in head.parameters():
            head_param_ids.add(id(p))

    backbone_param_ids: set[int] = set()
    global_branch = getattr(getattr(model, "backbone", None), "global_branch", None)
    if global_branch is not None:
        transformer = getattr(global_branch, "transformer", None)
        if transformer is not None:
            for p in transformer.parameters():
                backbone_param_ids.add(id(p))





    for _, p in model.named_parameters():
        if id(p) in backbone_param_ids:
            params_backbone.append(p)
        elif id(p) in head_param_ids:
            params_head.append(p)
        else:
            params_others.append(p)

    head_wd = float(weight_decay) * float(head_weight_decay_mult)
    param_groups = [
        {"params": params_others, "lr": base_lr, "weight_decay": weight_decay},
        {"params": params_backbone, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay},
    ]
    if params_head:
        param_groups.append({"params": params_head, "lr": base_lr, "weight_decay": head_wd})


    return torch.optim.AdamW(param_groups, weight_decay=0.0)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    if args.resume_from is not None and args.init_from is not None:
        raise ValueError(
            "--resume-from and --init-from are mutually exclusive.\n"
            "  --resume-from: Resume training (restores optimizer/scheduler/epoch).\n"
            "  --init-from: Initialize model for finetuning (fresh optimizer/scheduler, epoch=1)."
        )

    if args.reinit_head and args.init_from is None:
        print("[DINO] --reinit-head ignored because --init-from is not set.")

    if args.label_smoothing and args.label_smoothing > 0:
        print(f"[DINO] Label smoothing enabled: {args.label_smoothing}")

    if args.npz_info:
        if not args.npz_path:
            raise ValueError("--npz-info requires --npz-path")
        npz_path = Path(args.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        arrays = np.load(npz_path, allow_pickle=False)
        print(f"[DINO] NPZ: {npz_path}")
        print(f"[DINO]   images: {arrays['images'].shape}")
        print(f"[DINO]   writer_ids: {arrays['writer_ids'].shape}")
        print(f"[DINO]   is_forged: {arrays['is_forged'].shape}")
        if 'paths' in arrays:
            print(f"[DINO]   paths: {arrays['paths'].shape}")
        return



    dino_ckpt = Path(args.dino_ckpt) if args.dino_ckpt else None
    needs_backbone_ckpt = args.backbone_type in {"dino_vit", "cross_vit"} and not (
        args.init_from is not None and not args.always_load_dino
    )
    if needs_backbone_ckpt:
        if dino_ckpt is None:
            raise ValueError(f"--dino-ckpt is required when --backbone-type={args.backbone_type}")
        if not dino_ckpt.exists():
            raise FileNotFoundError(f"DINO checkpoint not found: {dino_ckpt}")
    elif dino_ckpt is not None and not dino_ckpt.exists():
        print(
            f"[DINO] Warning: --dino-ckpt does not exist and will be ignored for "
            f"backbone_type={args.backbone_type}: {dino_ckpt}"
        )


    if args.run_name:
        run_name = args.run_name
    else:
        run_name = input("请输入本次训练的名称: ")
    run_name = _sanitize_run_name(run_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{run_name}"
    training_start_time = datetime.now()

    if args.no_aug:
        train_transform = T.Compose([T.Resize((args.img_height, args.img_width)), T.ToTensor()])
    else:
        train_transform = build_train_transforms(
            args.img_height,
            args.img_width,
            use_thickness_aug=args.use_thickness_aug,
            thickness_aug_p=args.thickness_aug_p,
            use_elastic=args.use_elastic,
            elastic_alpha=args.elastic_alpha,
            elastic_sigma=args.elastic_sigma,
            elastic_p=args.elastic_p,
        )
    val_transform = build_val_transforms(args.img_height, args.img_width)


    partitions: Optional[Dict[str, List[str]]] = None
    if args.partitions_json:
        partitions_path = Path(args.partitions_json)
        if not partitions_path.exists():
            raise FileNotFoundError(f"Partitions JSON not found: {partitions_path}")
        with partitions_path.open("r", encoding="utf-8") as f:
            partitions = json.load(f)
        if "train" not in partitions or "val" not in partitions:
            raise ValueError("Partitions JSON must contain 'train' and 'val' keys.")
        print(f"[DINO] Using external partitions from {partitions_path}")
        print(f"[DINO]   train writers: {len(partitions['train'])}, val writers: {len(partitions['val'])}")

    if args.dataset_format == "chhd":
        if args.npz_path:
            raise ValueError("--dataset-format chhd does not support --npz-path")
        if args.match_by_sig:
            raise ValueError(
                "--match-by-sig is not supported with --dataset-format chhd; "
                "CHHD filename tokens like 01/02/03 are sample-set ids, not content groups."
            )
        if args.use_forg_head:
            raise ValueError("--use-forg-head is not supported with --dataset-format chhd because CHHD has no forged samples")
        print("[DINO] CHHD mode enabled: flat-folder genuine-only dataset with cross-writer negative pairs.")
        print("[DINO] CHHD metrics note: FAR/FRR/EER here mean same-writer vs different-writer errors, not skilled-forgery errors.")
    elif args.npz_path and args.match_by_sig:
        raise ValueError("--match-by-sig is not supported with --npz-path (no sig_id metadata)")
    if args.forged_subsets is not None:
        if args.npz_path:
            raise ValueError("--forged-subsets is only supported for standard folder datasets, not --npz-path")
        if args.dataset_format != "standard":
            raise ValueError("--forged-subsets is only supported with --dataset-format standard")
        print(f"[DINO] Forged subset filter enabled: {', '.join(args.forged_subsets)}")


    val_positive_pairs = args.val_positive_pairs if args.val_positive_pairs is not None else args.train_positive_pairs
    val_negative_pairs = args.val_negative_pairs if args.val_negative_pairs is not None else args.train_negative_pairs

    input_channels = int(args.input_channels) if args.input_channels is not None else (3 if args.npz_path else 1)

    train_ds, val_ds = build_datasets(
        args.data_root,
        train_transform=train_transform,
        val_transform=val_transform,
        writer_split_ratio=args.writer_split_ratio,
        split_seed=args.split_seed,
        train_positive_pairs=args.train_positive_pairs,
        train_negative_pairs=args.train_negative_pairs,
        val_positive_pairs=val_positive_pairs,
        val_negative_pairs=val_negative_pairs,
        pair_sampling_seed=args.pair_sampling_seed,
        balance_pairs=not args.no_balance_pairs,
        oversample=args.oversample,
        partitions=partitions,
        match_by_sig=args.match_by_sig,
        npz_path=args.npz_path,
        input_channels=input_channels,
        dataset_format=args.dataset_format,
        forged_subsets=args.forged_subsets,
    )
    summarize_train_pairs(train_ds)
    base_pair_sampling_seed = int(args.pair_sampling_seed) if args.pair_sampling_seed is not None else int(args.split_seed)
    if args.resample_train_pairs_each_epoch:
        print(
            "[DINO] Epoch-wise train pair resampling enabled: "
            f"base_seed={base_pair_sampling_seed}, epoch_seed=base_seed+(epoch-1)"
        )
    else:
        print("[DINO] Epoch-wise train pair resampling disabled: using fixed train pairs.")
    train_loader, val_loader = build_dataloaders(
        train_ds,
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )





    initial_freeze = bool(args.freeze_backbone) or int(getattr(args, "stage1_freeze_epochs", 0)) > 0



    if args.backbone_type == "timm_vit":
        dino_pretrained_path = None
        print(
            "[DINO] Using timm ViT backbone: "
            f"{args.backbone_model_name or 'vit_small_patch16_224.augreg_in21k_ft_in1k'}"
        )
    elif args.init_from is not None and not args.always_load_dino:
        dino_pretrained_path = None
        print(
            f"[DINO] --init-from specified: skipping DINO backbone weight loading "
            f"(will load full model from '{args.init_from}')"
        )
    else:
        dino_pretrained_path = str(dino_ckpt)
        if args.init_from is not None:
            print("[DINO] --always-load-dino enabled: loading DINO backbone before init-from")

    transformer_cfg = GlobalTransformerConfig(
        backbone_type=args.backbone_type,
        pretrained_path=dino_pretrained_path,
        freeze=initial_freeze,
    )
    if args.backbone_type == "cross_vit":
        transformer_cfg.model_name = args.backbone_model_name or "crossvit_15_240"
        transformer_cfg.timm_pretrained = bool(args.timm_pretrained) if args.timm_pretrained is not None else False
    elif args.backbone_type == "timm_vit":
        transformer_cfg.model_name = args.backbone_model_name or "vit_small_patch16_224.augreg_in21k_ft_in1k"
        transformer_cfg.timm_pretrained = bool(args.timm_pretrained) if args.timm_pretrained is not None else True
    backbone_cfg = HybridBackboneConfig(
        transformer=transformer_cfg,
        use_local_branch=args.use_local_branch,
        use_global_branch=args.use_global_branch,
    )
    backbone_cfg.in_channels = input_channels
    if args.npz_path and args.input_channels is None:
        print("[DINO] NPZ mode enabled: defaulting input_channels=3 (RGB)")
    backbone_cfg.fusion_dropout = float(args.dropout_rate)
    backbone_cfg.fusion_identity = bool(getattr(args, "fusion_identity", False))
    backbone_cfg.local_num_blocks = LOCAL_NUM_BLOCKS
    backbone_cfg.local_use_constant_channel = LOCAL_USE_CONSTANT_CHANNEL
    backbone_cfg.local_constant_channels = LOCAL_CONSTANT_CHANNELS
    backbone_cfg.local_use_sda = bool(args.use_sda)
    backbone_cfg.local_sda_groups = LOCAL_SDA_GROUPS
    backbone_cfg.local_use_sda_last_block = bool(args.sda_last_block)
    backbone_cfg.local_use_channel_spatial_attn = bool(args.use_channel_spatial_attn)
    backbone_cfg.local_csam_ratio = int(args.csam_ratio)
    backbone_cfg.local_use_lrsa = bool(args.use_lrsa)
    backbone_cfg.local_lrsa_num_heads = int(args.lrsa_num_heads)
    backbone_cfg.local_lrsa_q_pooled_size = int(args.lrsa_q_pooled_size)
    backbone_cfg.local_use_msc_fusion = bool(args.use_msc_fusion)
    backbone_cfg.local_msc_low_layer_idx = int(args.msc_low_layer_idx)
    backbone_cfg.local_msc_num_heads = int(args.msc_num_heads)

    backbone_cfg.local_use_ssp = bool(args.use_ssp)
    backbone_cfg.local_ssp_blocks = int(args.ssp_blocks)

    backbone_cfg.local_use_fsc = bool(args.use_fsc)
    local_input_height = int(args.local_input_height) if args.local_input_height is not None else int(args.img_height)
    local_input_width = int(args.local_input_width) if args.local_input_width is not None else int(args.img_width)
    backbone_cfg.local_input_height = local_input_height
    backbone_cfg.local_input_width = local_input_width
    model_cfg = HybridSiameseConfig(
        backbone=backbone_cfg,
        use_light_head=bool(args.use_light_head),
        light_head_hidden_dim=int(args.light_head_hidden_dim),
        light_head_dropout=float(args.light_head_dropout),
        use_forg_head=args.use_forg_head,
        use_writer_cls=args.use_writer_cls,
        writer_cls_num_classes=train_ds.num_writers if args.use_writer_cls else 0,
    )

    model: torch.nn.Module = HybridSiameseNet(model_cfg).to(device)

    if args.compile:
        model = cast(torch.nn.Module, torch.compile(model))


    print_model_summary(model, "DINO Siamese Network")

    loss_cfg = LossConfig(pairwise_type="contrastive", margin=args.margin)


    print(f"[DINO] Using loss type: {args.loss_type}")
    if args.loss_type == "hard_mining":
        print(f"[DINO] Hard Negative Mining - neg_ratio={args.hard_neg_ratio}, pos_ratio={args.hard_pos_ratio}")
        def loss_fn(feat_a: torch.Tensor, feat_b: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
            return contrastive_loss_with_hard_mining(
                feat_a,
                feat_b,
                same,
                margin=loss_cfg.margin,
                a=1.0,
                beta=1.0,
                hard_neg_ratio=args.hard_neg_ratio,
                hard_pos_ratio=args.hard_pos_ratio,
            )
    elif args.loss_type == "focal":
        _ls = args.label_smoothing
        print(f"[DINO] Focal Contrastive Loss - gamma={args.focal_gamma}, label_smoothing={_ls}")
        def loss_fn(feat_a: torch.Tensor, feat_b: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
            _same = same
            if _ls > 0:
                _same = same.float() * (1.0 - 2 * _ls) + _ls
            return focal_contrastive_loss(
                feat_a,
                feat_b,
                _same,
                margin=loss_cfg.margin,
                gamma=args.focal_gamma,
                a=1.0,
                beta=1.0,
            )
    elif args.loss_type == "bce_distance":
        print(f"[DINO] BCE Distance Loss - margin={args.margin} (no saturation, directly optimizes classification boundary)")
        def loss_fn(feat_a: torch.Tensor, feat_b: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
            return bce_distance_loss(
                feat_a,
                feat_b,
                same,
                margin=loss_cfg.margin,
            )
    elif args.loss_type == "smooth":
        print(f"[DINO] Smooth Contrastive Loss - margin={args.margin} (no saturation, exponential penalty for close negatives)")
        def loss_fn(feat_a: torch.Tensor, feat_b: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
            return smooth_contrastive_loss(
                feat_a,
                feat_b,
                same,
                margin=loss_cfg.margin,
                temperature=0.5,
            )
    elif args.loss_type == "bce_logits":


        pos_weight = torch.tensor([args.bce_pos_weight], device=device) if args.bce_pos_weight != 1.0 else None
        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[DINO] BCEWithLogitsLoss on head logits (pos_weight={args.bce_pos_weight})")

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return bce_criterion(logits.view(-1), labels.float())
    else:

        def loss_fn(feat_a: torch.Tensor, feat_b: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
            return contrastive_loss_signature_verification(
                feat_a,
                feat_b,
                same,
                margin=loss_cfg.margin,
                a=1.0,
                beta=1.0,
            )


    use_logits_loss = (args.loss_type == "bce_logits")


    if args.use_forg_head:
        print(f"[DINO] Forgery detection head enabled: λ={args.forg_lambda} (HTCSigNet style)")
    else:
        print("[DINO] Forgery detection head disabled (baseline training)")

    if args.use_writer_cls:
        print(f"[DINO] Writer classification head enabled: λ={args.writer_cls_lambda}, num_classes={train_ds.num_writers}")
    else:
        print("[DINO] Writer classification head disabled")


    backbone_lr_mult = args.backbone_lr_mult if not args.freeze_backbone else 0.0
    optimizer = build_optimizer_dino(
        model,
        args.base_lr,
        backbone_lr_mult,
        args.weight_decay,
        head_weight_decay_mult=args.head_weight_decay_mult,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler_step_mismatch_warned = False

    if getattr(args, 'exponential_lr_gamma', None) is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exponential_lr_gamma)
        step_scheduler = None
        print(f"[DINO] Using epoch-based ExponentialLR scheduler (gamma={args.exponential_lr_gamma})")
    elif args.use_epoch_scheduler:





        _min_lr_ratio = args.min_lr_ratio
        _T_max = args.epochs
        def _cosine_epoch_lambda(epoch: int) -> float:
            if epoch >= _T_max:
                return _min_lr_ratio
            progress = epoch / _T_max
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return _min_lr_ratio + (1.0 - _min_lr_ratio) * cosine_decay
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_epoch_lambda)
        step_scheduler = None
        print(f"[DINO] Using epoch-based LambdaLR cosine scheduler (min_lr_ratio={_min_lr_ratio})")
    else:
        warmup_steps = int(args.warmup_epochs * steps_per_epoch) if args.warmup_epochs > 0 else 0
        if args.warmup_ratio > 0:
            warmup_steps = int(total_steps * args.warmup_ratio)

        step_scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        scheduler = None
        print(f"[DINO] Using step-based scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, min_lr_ratio={args.min_lr_ratio}")

    scaler = _create_grad_scaler(device.type, enabled=args.mixed_precision and device.type == "cuda")

    start_epoch = 1
    training_end_time = None
    best_acc_best: float | None = None
    best_auc = None
    best_loss = float("inf")
    best_val_loss_ckpt = float("inf")
    best_snapshot: dict | None = None
    epochs_without_improvement = 0
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_best_loss{ckpt_path.suffix}")


    logs_root = Path("logs")
    logs_root.mkdir(parents=True, exist_ok=True)
    logs_dir = logs_root / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_dir / "train.log"


    command_path = logs_dir / "command.txt"
    command_str = "python " + " ".join(sys.argv)
    with command_path.open("w", encoding="utf-8") as f:
        f.write(f"Command used for this run:\n{command_str}\n")


    tf_logs_root = Path(args.tf_logs_root)
    tf_logs_root.mkdir(parents=True, exist_ok=True)
    tf_logs_dir = tf_logs_root / run_id
    tf_logs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tf_logs_dir))


    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint '{resume_path}' not found.")
        checkpoint = torch.load(resume_path, map_location=device)
        _model_base = getattr(model, "_orig_mod", model)
        _model_base.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "step_scheduler" in checkpoint and step_scheduler is not None:
            step_scheduler.load_state_dict(checkpoint["step_scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_acc_best = checkpoint.get("best_acc_best", None)
        best_auc = checkpoint.get("best_auc", None)
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        best_val_loss_ckpt = float(checkpoint.get("best_val_loss_ckpt", best_loss))
        print(f"[DINO] Resumed training from checkpoint '{resume_path}' at epoch {start_epoch}.")


    elif args.init_from is not None:
        init_path = Path(args.init_from)
        if not init_path.is_file():
            raise FileNotFoundError(f"Init checkpoint '{init_path}' not found.")
        checkpoint = torch.load(init_path, map_location=device)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint


        _model_base = getattr(model, "_orig_mod", model)
        if args.reinit_head:
            head_prefixes = ("head.", "forg_head.", "cls_head.")
            filtered_state = {k: v for k, v in state_dict.items() if not k.startswith(head_prefixes)}

            _current_sd = _model_base.state_dict()
            _shape_skipped = []
            _compatible_state = {}
            for k, v in filtered_state.items():
                if k in _current_sd and _current_sd[k].shape != v.shape:
                    _shape_skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(_current_sd[k].shape)}")
                else:
                    _compatible_state[k] = v
            if _shape_skipped:
                print(f"[DINO] --init-from: Skipped {len(_shape_skipped)} shape-mismatched keys:")
                for s in _shape_skipped:
                    print(f"  {s}")
            missing, unexpected = _model_base.load_state_dict(_compatible_state, strict=False)

            if args.init_strict:
                def _is_head_key(key: str) -> bool:
                    return key.startswith(head_prefixes)

                missing_non_head = [k for k in missing if not _is_head_key(k)]
                unexpected_non_head = [k for k in unexpected if not _is_head_key(k)]
                if missing_non_head or unexpected_non_head:
                    raise RuntimeError(
                        "[DINO] --init-from strict load failed with --reinit-head. "
                        f"Missing: {missing_non_head} Unexpected: {unexpected_non_head}"
                    )

            if missing:
                print(f"[DINO] --init-from (reinit-head): Missing keys: {missing}")
            if unexpected:
                print(f"[DINO] --init-from (reinit-head): Unexpected keys: {unexpected}")

            def _reset_module_params(module: torch.nn.Module) -> int:
                reset_count = 0
                for sub in module.modules():
                    reset_fn = getattr(sub, "reset_parameters", None)
                    if callable(reset_fn):
                        reset_fn()
                        reset_count += 1
                return reset_count

            head_reset = _reset_module_params(_model_base.head)
            forg_reset = 0
            if getattr(_model_base, "forg_head", None) is not None:
                forg_reset = _reset_module_params(_model_base.forg_head)

            dropped_keys = [k for k in state_dict.keys() if k.startswith(head_prefixes)]
            print(
                f"[DINO] --reinit-head: dropped {len(dropped_keys)} head keys, "
                f"reinitialized head modules={head_reset}, forg_head modules={forg_reset}."
            )
            print(
                f"[DINO] Initialized model from '{init_path}' for finetuning "
                "(optimizer/scheduler/epoch reset, start_epoch=1)."
            )
        else:

            _current_sd = _model_base.state_dict()
            _shape_skipped = []
            _compatible_state = {}
            for k, v in state_dict.items():
                if k in _current_sd and _current_sd[k].shape != v.shape:
                    _shape_skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(_current_sd[k].shape)}")
                else:
                    _compatible_state[k] = v
            if _shape_skipped:
                print(f"[DINO] --init-from: Skipped {len(_shape_skipped)} shape-mismatched keys:")
                for s in _shape_skipped:
                    print(f"  {s}")
            missing, unexpected = _model_base.load_state_dict(_compatible_state, strict=args.init_strict)
            if missing:
                print(f"[DINO] --init-from: Missing keys: {missing}")
            if unexpected:
                print(f"[DINO] --init-from: Unexpected keys: {unexpected}")
            print(f"[DINO] Initialized model from '{init_path}' for finetuning (optimizer/scheduler/epoch reset, start_epoch=1).")

    def _set_dino_backbone_trainable(trainable: bool) -> bool:

        try:
            _base = getattr(model, "_orig_mod", model)
            if trainable:
                return _base.stage2_unfreeze_backbone()
            else:
                return _base.stage1_freeze()
        except AttributeError:
            return False


    if (not args.freeze_backbone) and int(getattr(args, "stage1_freeze_epochs", 0)) > 0:
        if start_epoch > int(args.stage1_freeze_epochs):
            ok = _set_dino_backbone_trainable(True)
            if ok:
                print(f"[DINO] Stage2: backbone unfrozen (resume at epoch {start_epoch}).")

    stage1_freeze_epochs = int(getattr(args, "stage1_freeze_epochs", 0))
    if stage1_freeze_epochs < 0:
        raise ValueError("--stage1-freeze-epochs must be >= 0")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.resample_train_pairs_each_epoch:

            epoch_pair_seed = refresh_train_pairs_for_epoch(train_ds, epoch, base_pair_sampling_seed)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            print(f"[DINO] Epoch {epoch}: resampled train pairs with seed={epoch_pair_seed}")
            summarize_train_pairs(train_ds)
            if (not args.use_epoch_scheduler) and (len(train_loader) != steps_per_epoch) and (not scheduler_step_mismatch_warned):
                print(
                    "[DINO] Warning: train steps-per-epoch changed after resampling "
                    f"({steps_per_epoch} -> {len(train_loader)}). "
                    "Step-based scheduler total_steps was computed from initial loader length."
                )
                scheduler_step_mismatch_warned = True



        if (
            (not args.freeze_backbone)
            and stage1_freeze_epochs > 0
            and epoch == stage1_freeze_epochs + 1
        ):
            ok = _set_dino_backbone_trainable(True)
            if ok:
                print(f"[DINO] Stage2: backbone unfrozen at epoch {epoch} (after {stage1_freeze_epochs} frozen epochs).")
            else:
                print("[DINO] Stage2: requested unfreeze but backbone transformer was not found.")

        train_loss, train_acc, train_acc_at_05, train_forg_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler,
            use_amp=args.mixed_precision,
            amp_dtype=args.amp_dtype,
            max_grad_norm=args.max_grad_norm,
            scheduler=step_scheduler,
            writer=writer,
            epoch=epoch,
            use_logits_loss=use_logits_loss,
            use_forg_head=args.use_forg_head,
            forg_lambda=args.forg_lambda,
            use_writer_cls=args.use_writer_cls,
            writer_cls_lambda=args.writer_cls_lambda,
        )
        val_loss, val_auc, val_acc, val_acc_best, val_acc_at_05, val_frr, val_far, val_eer, val_thr, val_thr_best, val_frr_best, val_far_best, val_eer_at_best_acc, val_far_at_05, val_frr_at_05, fa_count, fr_count, fa_items, fr_items = validate(
            model, val_loader, loss_fn, device, use_amp=args.mixed_precision, amp_dtype=args.amp_dtype, error_topk=args.error_topk, use_logits_loss=use_logits_loss
        )

        if scheduler is not None:
            scheduler.step()

        lr_display = format_lrs(optimizer)



        acc_at_05_str = f"{val_acc_at_05:.4f}" if not math.isnan(val_acc_at_05) else "N/A"
        far_at_05_str = f"{val_far_at_05:.4f}" if not math.isnan(val_far_at_05) else "N/A"
        frr_at_05_str = f"{val_frr_at_05:.4f}" if not math.isnan(val_frr_at_05) else "N/A"
        train_acc_at_05_str = f"{train_acc_at_05:.4f}" if not math.isnan(train_acc_at_05) else "N/A"


        if not (math.isnan(val_far_at_05) or math.isnan(val_frr_at_05)):
            mean_err_at_05 = (val_far_at_05 + val_frr_at_05) / 2
            mean_err_at_05_str = f"{mean_err_at_05:.4f}"
        else:
            mean_err_at_05_str = "N/A"


        table_lines = [
            f"",
            f"[DINO] Epoch {epoch}/{args.epochs}  |  LR: {lr_display}",
            f"  Train: loss={train_loss:.4f}  acc={train_acc:.4f}  acc@0.5={train_acc_at_05_str}  forg_loss={train_forg_loss:.4f}",
            f"  Valid: loss={val_loss:.4f}  auc={val_auc:.4f}",
            f"",
            f"  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐",
            f"  │   Metric    │   EER Threshold  │ Best ACC Thresh  │ Fixed (prob=0.5) │",
            f"  ├─────────────┼──────────────────┼──────────────────┼──────────────────┤",
            f"  │  Threshold  │      {val_thr:>7.4f}     │      {val_thr_best:>7.4f}     │      0.0000      │",
            f"  │  FAR        │      {val_far:>7.4f}     │      {val_far_best:>7.4f}     │      {far_at_05_str:>6}     │",
            f"  │  FRR        │      {val_frr:>7.4f}     │      {val_frr_best:>7.4f}     │      {frr_at_05_str:>6}     │",
            f"  │  ACC        │      {val_acc:>7.4f}     │      {val_acc_best:>7.4f}     │      {acc_at_05_str:>6}     │",
            f"  │  Mean Err   │      {val_eer:>7.4f}     │      {val_eer_at_best_acc:>7.4f}     │      {mean_err_at_05_str:>6}     │",
            f"  └─────────────┴──────────────────┴──────────────────┴──────────────────┘",
            f"",
        ]
        for line in table_lines:
            print(line)
            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


        global_step = epoch
        writer.add_scalar("loss/train", train_loss, global_step)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("loss/train_forg", train_forg_loss, global_step)
        writer.add_scalar("metrics/train_acc", train_acc, global_step)
        writer.add_scalar("metrics/auc", val_auc, global_step)
        writer.add_scalar("metrics/acc_eer", val_acc, global_step)
        writer.add_scalar("metrics/acc_best", val_acc_best, global_step)
        writer.add_scalar("metrics/frr", val_frr, global_step)
        writer.add_scalar("metrics/far", val_far, global_step)
        writer.add_scalar("metrics/eer", val_eer, global_step)
        writer.add_scalar("metrics/thr_eer", val_thr, global_step)
        writer.add_scalar("metrics/thr_best", val_thr_best, global_step)
        writer.add_scalar("metrics/frr_best", val_frr_best, global_step)
        writer.add_scalar("metrics/far_best", val_far_best, global_step)
        writer.add_scalar("metrics/eer_at_best_acc", val_eer_at_best_acc, global_step)
        writer.add_scalar("metrics/acc_at_05", val_acc_at_05, global_step)
        writer.add_scalar("metrics/far_at_05", val_far_at_05, global_step)
        writer.add_scalar("metrics/frr_at_05", val_frr_at_05, global_step)
        writer.add_scalar("errors/fa_count", fa_count, global_step)
        writer.add_scalar("errors/fr_count", fr_count, global_step)


        if fa_items:
            fa_text_lines = [f"FA#{item.rank}: score={item.score:.4f} GT={item.gt_text} Pred={item.pred_text} path_a={item.path_a} path_b={item.path_b}" for item in fa_items]
            writer.add_text(
                f"errors/{run_name}/FA/text",
                f"**Epoch {epoch}**\n" + "\n".join(fa_text_lines),
                global_step,
            )
        if fr_items:
            fr_text_lines = [f"FR#{item.rank}: score={item.score:.4f} GT={item.gt_text} Pred={item.pred_text} path_a={item.path_a} path_b={item.path_b}" for item in fr_items]
            writer.add_text(
                f"errors/{run_name}/FR/text",
                f"**Epoch {epoch}**\n" + "\n".join(fr_text_lines),
                global_step,
            )


        for item in fa_items:
            img_tensor = create_annotated_error_image(item)
            if img_tensor is not None:
                writer.add_image(f"errors/{run_name}/FA/top{item.rank:02d}", img_tensor, global_step)

        for item in fr_items:
            img_tensor = create_annotated_error_image(item)
            if img_tensor is not None:
                writer.add_image(f"errors/{run_name}/FR/top{item.rank:02d}", img_tensor, global_step)

        writer.add_scalar("lr/base", optimizer.param_groups[0]["lr"], global_step)



        if not math.isnan(val_auc):
            if best_auc is None or val_auc > best_auc:
                best_auc = val_auc
        if val_loss < best_loss:
            best_loss = val_loss


        save_best = False
        if not math.isnan(val_acc_best):
            if best_acc_best is None or val_acc_best > best_acc_best:
                best_acc_best = val_acc_best
                save_best = True


        save_best_loss = False
        if not math.isnan(val_loss) and val_loss < best_val_loss_ckpt:
            best_val_loss_ckpt = val_loss
            save_best_loss = True
            loss_snapshot = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "step_scheduler": step_scheduler.state_dict() if step_scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "best_acc_best": best_acc_best,
                "best_auc": best_auc,
                "best_loss": best_loss,
                "best_val_loss_ckpt": best_val_loss_ckpt,
            }
            torch.save(loss_snapshot, best_loss_ckpt_path)
            best_loss_msg = (
                f"[DINO] Saved best-loss checkpoint to {best_loss_ckpt_path} "
                f"(epoch={epoch}, best_val_loss={best_val_loss_ckpt:.4f})"
            )
            print(best_loss_msg)
            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(best_loss_msg + "\n")
            writer.add_scalar("best/best_val_loss_ckpt", best_val_loss_ckpt, epoch)


        if save_best:
            epochs_without_improvement = 0
            best_snapshot = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "step_scheduler": step_scheduler.state_dict() if step_scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "best_acc_best": best_acc_best,
                "best_auc": best_auc,
                "best_loss": best_loss,
                "best_val_loss_ckpt": best_val_loss_ckpt,
            }
            torch.save(best_snapshot, ckpt_path)
            best_acc_display = f"{best_acc_best:.4f}" if best_acc_best is not None else "N/A"
            best_auc_display = f"{best_auc:.4f}" if best_auc is not None else "N/A"
            best_msg = (
                f"[DINO] Saved best checkpoint to {ckpt_path} "
                f"(epoch={epoch}, best_acc_best={best_acc_display}, best_auc={best_auc_display}, best_loss={best_loss:.4f})"
            )
            print(best_msg)
            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(best_msg + "\n")
            writer.add_scalar("best/best_acc_best", float(best_acc_best) if best_acc_best is not None else float("nan"), epoch)
            writer.add_scalar("best/best_auc", float(best_auc) if best_auc is not None else float("nan"), epoch)
            writer.add_scalar("best/best_loss", best_loss, epoch)
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                early_stop_msg = f"[DINO] Early stopping triggered after {epochs_without_improvement} epochs without improvement (patience={args.early_stopping_patience})"
                print(early_stop_msg)
                with run_log_path.open("a", encoding="utf-8") as f:
                    f.write(early_stop_msg + "\n")
                training_end_time = datetime.now()
                break

    writer.close()

    if training_end_time is None:
        training_end_time = datetime.now()

    duration = training_end_time - training_start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}h {minutes}m {seconds}s"

    print(f"\n[DINO] Training completed in {duration_str}")
    best_acc_display = f"{best_acc_best:.4f}" if best_acc_best is not None else "N/A"
    best_auc_display = f"{best_auc:.4f}" if best_auc is not None else "N/A"
    print(f"[DINO] Best metrics: ACC={best_acc_display}, AUC={best_auc_display}, Loss={best_loss:.4f}")

    return {
        "run_name": run_name,
        "run_id": run_id,
        "start_time": training_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": training_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_str": duration_str,
        "best_acc_best": best_acc_best,
        "best_auc": best_auc,
        "best_loss": best_loss,
        "epochs_completed": epoch if 'epoch' in locals() else 0,
        "total_epochs": args.epochs,
    }


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        error_msg = traceback.format_exc()
        print(f"\n[ERROR] Training failed with exception:\n{error_msg}")
        sys.exit(1)
