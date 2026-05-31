"""Composite loss utilities for hybrid Siamese training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    lambda_pair: float = 1.0
    lambda_id: float = 0.0
    lambda_forg: float = 0.3
    pairwise_type: str = "bce"                    
    margin: float = 1.0
                               
    use_hard_mining: bool = False
    hard_neg_ratio: float = 0.5               
    hard_pos_ratio: float = 0.3               


def contrastive_loss_signature_verification(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
    margin: float = 1.0,
    a: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """欧氏距离对比损失，用于离线手写签名验证。

    论文公式：
        L(s1, s2, y) = a * (1 - y) * D^2 + beta * y * max(0, m - D)^2

    其中：
        - y = 1 表示同一作者（genuine pair）；
        - y = 0 表示不同作者或含伪造（forgery pair）；
        - D 为两特征向量的欧氏距离；
        - m 为 margin。

    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量，取值为 0 或 1。
        margin:    对比损失的 margin（论文中 m = 1）。
        a:         genuine 对的权重常数（论文中 a = 1）。
        beta:      forgery 对的权重常数（论文中 beta = 1）。

    Returns:
        标量张量，对整个 batch 的平均损失。
    """

                                          
    feature_s1 = feature_s1.float()
    feature_s2 = feature_s2.float()
    label_y = label_y.float().to(feature_s1.device)

                                 
                                                                       
    D = torch.nn.functional.pairwise_distance(feature_s1, feature_s2, p=2)

                            
    genuine_loss = a * label_y * (D ** 2)

                                        
                            
    forgery_term = torch.relu(margin - D)
    forgery_loss = beta * (1.0 - label_y) * (forgery_term ** 2)

                   
    loss = (genuine_loss + forgery_loss).mean()
    return loss


def contrastive_loss_with_hard_mining(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
    margin: float = 1.0,
    a: float = 1.0,
    beta: float = 1.0,
    hard_neg_ratio: float = 0.5,
    hard_pos_ratio: float = 0.3,
) -> torch.Tensor:
    """带 Hard Negative Mining 的对比损失。
    
    Hard Negative Mining 策略：
    - 对于负样本（FA对）：选择距离最近（最难区分）的样本，给予更高权重
    - 对于正样本：选择距离最远（最难匹配）的样本，给予更高权重
    
    这样可以让模型更关注难以区分的 FA 对。

    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量，取值为 0 或 1。
        margin:    对比损失的 margin。
        a:         genuine 对的权重常数。
        beta:      forgery 对的权重常数。
        hard_neg_ratio: 被视为"困难负样本"的比例 (0-1)
        hard_pos_ratio: 被视为"困难正样本"的比例 (0-1)

    Returns:
        标量张量，对整个 batch 的加权平均损失。
    """
    feature_s1 = feature_s1.float()
    feature_s2 = feature_s2.float()
    label_y = label_y.float().to(feature_s1.device)

            
    D = F.pairwise_distance(feature_s1, feature_s2, p=2)

               
    pos_mask = (label_y == 1)
    neg_mask = (label_y == 0)
    
    pos_count = pos_mask.sum().item()
    neg_count = neg_mask.sum().item()

             
    sample_weights = torch.ones_like(label_y)
    
                                                 
    if neg_count > 1:
        neg_distances = D[neg_mask]
                                
        neg_ranks = neg_distances.argsort().argsort().float()
                     
        neg_difficulty = 1.0 - neg_ranks / (neg_count - 1 + 1e-8)
        
                                         
                                                   
        hard_threshold = 1.0 - hard_neg_ratio
        neg_weights = torch.where(
            neg_difficulty >= hard_threshold,
            1.0 + neg_difficulty,                     
            0.5 + 0.5 * neg_difficulty,                     
        )
        sample_weights[neg_mask] = neg_weights
    
                                                 
    if pos_count > 1:
        pos_distances = D[pos_mask]
                  
        pos_ranks = (-pos_distances).argsort().argsort().float()
        pos_difficulty = 1.0 - pos_ranks / (pos_count - 1 + 1e-8)
        
        hard_threshold = 1.0 - hard_pos_ratio
        pos_weights = torch.where(
            pos_difficulty >= hard_threshold,
            1.0 + 0.5 * pos_difficulty,                        
            0.8 + 0.2 * pos_difficulty,
        )
        sample_weights[pos_mask] = pos_weights

            
    genuine_loss = a * label_y * (D ** 2)
    forgery_term = torch.relu(margin - D)
    forgery_loss = beta * (1.0 - label_y) * (forgery_term ** 2)
    
    per_sample_loss = genuine_loss + forgery_loss
    weighted_loss = per_sample_loss * sample_weights
    
                    
    loss = weighted_loss.sum() / (sample_weights.sum() + 1e-8)
    
    return loss


def focal_contrastive_loss(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
    margin: float = 1.0,
    gamma: float = 2.0,
    a: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal 风格的对比损失 - 自动关注困难样本。
    
    借鉴 Focal Loss 的思想：对于模型已经能很好区分的样本降低权重，
    对于模型难以区分的样本增加权重。
    
    对于负样本（FA对）：如果距离已经很大（容易区分），降低权重
    对于正样本：如果距离已经很小（容易匹配），降低权重

    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量，取值为 0 或 1。
        margin:    对比损失的 margin。
        gamma:     focusing parameter，越大越关注困难样本
        a:         genuine 对的权重常数。
        beta:      forgery 对的权重常数。
        eps:       数值稳定性常数，防止 NaN。

    Returns:
        标量张量，Focal 加权的平均损失。
    """
    feature_s1 = feature_s1.float()
    feature_s2 = feature_s2.float()
    label_y = label_y.float().to(feature_s1.device)
    
                              
    if torch.isnan(feature_s1).any() or torch.isinf(feature_s1).any():
        feature_s1 = torch.nan_to_num(feature_s1, nan=0.0, posinf=1.0, neginf=-1.0)
    if torch.isnan(feature_s2).any() or torch.isinf(feature_s2).any():
        feature_s2 = torch.nan_to_num(feature_s2, nan=0.0, posinf=1.0, neginf=-1.0)

                          
    D = F.pairwise_distance(feature_s1, feature_s2, p=2, eps=eps)
    
                           
    D_clamped = torch.clamp(D, min=0.0, max=10.0)
    
                                           
                                                   
    
                       
    pos_easy = torch.exp(-D_clamped)                   
    
                       
    neg_margin_diff = torch.relu(D_clamped - margin * 0.5)
    neg_easy = 1.0 - torch.exp(-neg_margin_diff)           
    
                                                         
    pos_focal_weight = (1.0 - pos_easy + eps).pow(gamma)
    neg_focal_weight = (1.0 - neg_easy + eps).pow(gamma)
    
                              
    pos_focal_weight = torch.clamp(pos_focal_weight, min=eps, max=10.0)
    neg_focal_weight = torch.clamp(neg_focal_weight, min=eps, max=10.0)
    
            
    genuine_loss = a * label_y * (D ** 2) * pos_focal_weight
    forgery_term = torch.relu(margin - D)
    forgery_loss = beta * (1.0 - label_y) * (forgery_term ** 2) * neg_focal_weight

    loss = (genuine_loss + forgery_loss).mean()
    
                                  
    if torch.isnan(loss) or torch.isinf(loss):
        genuine_loss_simple = a * label_y * (D ** 2)
        forgery_loss_simple = beta * (1.0 - label_y) * (forgery_term ** 2)
        loss = (genuine_loss_simple + forgery_loss_simple).mean()
                            
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(margin, device=feature_s1.device, dtype=feature_s1.dtype)
    
    return loss


def smooth_contrastive_loss(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
    margin: float = 1.0,
    temperature: float = 0.5,
) -> torch.Tensor:
    """平滑对比损失 - 没有饱和区域，loss 始终与 accuracy 计算方式一致。
    
    与传统对比损失不同，这个损失函数：
    1. 对于负样本：使用 exp(-D) 而非 max(0, margin-D)²，距离越小损失越大，永不饱和
    2. 对于正样本：使用 D² 保持不变
    3. 添加 temperature 参数控制损失的敏感度
    
    这样可以避免 "val loss 低但 val acc 不高" 的问题，因为损失函数不存在饱和区。

    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量，1=genuine pair, 0=forgery pair。
        margin:    用于缩放负样本损失的参考距离。
        temperature: 温度参数，控制负样本损失的敏感度，越小越敏感。

    Returns:
        标量张量，平均损失。
    """
    feature_s1 = feature_s1.float()
    feature_s2 = feature_s2.float()
    label_y = label_y.float().to(feature_s1.device)

    D = F.pairwise_distance(feature_s1, feature_s2, p=2)
    
                                        
    genuine_loss = label_y * (D ** 2)
    
                                          
                                                         
                               
                                   
    forgery_loss = (1.0 - label_y) * (margin ** 2) * torch.exp(-D / temperature)
    
    loss = (genuine_loss + forgery_loss).mean()
    return loss


def bce_distance_loss(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """基于距离的 BCE Loss - 将距离转换为相似度概率后使用 BCE。
    
    这个损失函数直接优化分类边界，使得 loss 和 accuracy 的计算方式一致：
    1. 计算特征对之间的欧氏距离 D
    2. 将距离转换为相似度分数: similarity = 1 / (1 + D/margin)
    3. 使用 BCE loss 优化这个相似度分数
    
    优点：
    - Loss 和 accuracy 使用相同的"分数"计算方式
    - 没有 margin 饱和问题
    - 梯度更平滑

    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量，1=genuine pair, 0=forgery pair。
        margin:    距离缩放因子，控制距离到概率的映射。

    Returns:
        标量张量，BCE 损失。
    """
    feature_s1 = feature_s1.float()
    feature_s2 = feature_s2.float()
    label_y = label_y.float().to(feature_s1.device)

    D = F.pairwise_distance(feature_s1, feature_s2, p=2)
    
                        
                            
                                   
                              
    similarity = 1.0 / (1.0 + D / margin)
    
                 
                      
    similarity = torch.clamp(similarity, min=1e-7, max=1.0 - 1e-7)
    
    loss = F.binary_cross_entropy(similarity, label_y)
    return loss


def cosine_bce_loss(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
) -> torch.Tensor:
    """基于余弦相似度的 BCE Loss。
    
    使用余弦相似度而非欧氏距离，然后应用 BCE loss。
    余弦相似度天然在 [-1, 1] 范围内，转换到 [0, 1] 后直接用 BCE。

    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量，1=genuine pair, 0=forgery pair。

    Returns:
        标量张量，BCE 损失。
    """
    feature_s1 = feature_s1.float()
    feature_s2 = feature_s2.float()
    label_y = label_y.float().to(feature_s1.device)

                     
    cos_sim = F.cosine_similarity(feature_s1, feature_s2, dim=-1)
    
                                   
    similarity = (cos_sim + 1.0) / 2.0
    
           
    similarity = torch.clamp(similarity, min=1e-7, max=1.0 - 1e-7)
    
    loss = F.binary_cross_entropy(similarity, label_y)
    return loss


def combined_contrastive_bce_loss(
    feature_s1: torch.Tensor,
    feature_s2: torch.Tensor,
    label_y: torch.Tensor,
    margin: float = 1.0,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """组合损失：对比损失 + BCE 距离损失。
    
    结合两种损失的优点：
    - 对比损失提供明确的 margin 约束
    - BCE 损失确保 loss 与 accuracy 计算一致
    
    Args:
        feature_s1: 形状 [B, D] 的特征张量。
        feature_s2: 形状 [B, D] 的特征张量。
        label_y:   形状 [B] 的标签张量。
        margin:    对比损失的 margin。
        bce_weight: BCE 损失的权重，范围 [0, 1]。

    Returns:
        标量张量，组合损失。
    """
    contrastive = contrastive_loss_signature_verification(
        feature_s1, feature_s2, label_y, margin=margin
    )
    bce = bce_distance_loss(feature_s1, feature_s2, label_y, margin=margin)
    
    return (1.0 - bce_weight) * contrastive + bce_weight * bce


class PairwiseLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pairwise_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = None

    def forward(self, logits_or_dist: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pairwise loss.

        - 当 pairwise_type == "bce" 时，输入被视为 logits，使用 BCEWithLogitsLoss。
        - 当 pairwise_type == "contrastive" 时，输入被视为欧氏距离标量 d，
          使用标准对比损失：
              L = y * d^2 + (1-y) * max(0, margin - d)^2
        """
        if self.cfg.pairwise_type == "bce":
            return self.loss_fn(logits_or_dist, labels.float())
        if self.cfg.pairwise_type == "contrastive":
            dist = logits_or_dist.float()
            labels_f = labels.float()
            pos = labels_f * dist.pow(2)
            neg = (1.0 - labels_f) * torch.relu(self.cfg.margin - dist).pow(2)
            return (pos + neg).mean()
        raise ValueError(f"Unknown pairwise loss {self.cfg.pairwise_type}")


class AuxiliaryHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(8, int(in_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompositeLoss(nn.Module):
    def __init__(self, cfg: LossConfig, num_writers: int, embed_dim: int, use_forgery_head: bool = False):
        super().__init__()
        self.cfg = cfg
        self.pairwise = PairwiseLoss(cfg)
        self.use_forgery = use_forgery_head
        self.id_head = AuxiliaryHead(in_dim=embed_dim, num_classes=num_writers)
        self.id_loss = nn.CrossEntropyLoss()
        if use_forgery_head:
            self.forg_head = AuxiliaryHead(in_dim=embed_dim, num_classes=2)
            self.forg_loss = nn.CrossEntropyLoss()
        else:
            self.forg_head = None

    def forward(
        self,
        pair_logits: torch.Tensor,
        pair_labels: torch.Tensor,
        fused: torch.Tensor,
        writer_ids: Optional[torch.Tensor] = None,
        forgery_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
                        
                                              
                                                    
        if self.cfg.pairwise_type == "contrastive":
                                                  
                                                
                                          
                          
                                                                            
            lp = self.pairwise(pair_logits, pair_labels)
        else:
            lp = self.pairwise(pair_logits, pair_labels)
        losses["pair"] = lp
        total = self.cfg.lambda_pair * lp
        if writer_ids is not None:
            logits_id = self.id_head(fused)
            lid = self.id_loss(logits_id, writer_ids)
            losses["id"] = lid
            total = total + self.cfg.lambda_id * lid
        if self.use_forgery and forgery_labels is not None:
            logits_forg = self.forg_head(fused)
            lfg = self.forg_loss(logits_forg, forgery_labels)
            losses["forg"] = lfg
            total = total + self.cfg.lambda_forg * lfg
        losses["total"] = total
        return losses
