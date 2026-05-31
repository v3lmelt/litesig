import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3,
                 max_spatial_size=28):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.max_spatial_size = max_spatial_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2

        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)


    def forward(self, x, y):
        B, C, H, W = x.shape
        original_h, original_w = H, W
        need_upsample = False

        if H > self.max_spatial_size or W > self.max_spatial_size:
            need_upsample = True
            scale_factor = self.max_spatial_size / max(H, W)
            new_h = int(H * scale_factor)
            new_w = int(W * scale_factor)
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(new_h, new_w), mode='bilinear', align_corners=False)
            H, W = new_h, new_w

        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y = y1 + y2 + y3

        y = y.flatten(-2, -1)
        y = y.transpose(1, 2)
        y = self.layer_norm(y)

        B, N1, C = y.shape
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        topk1 = max(1, int(N1 / self.k1))
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, dtype=x.dtype)
        index = torch.topk(attn, k=topk1, dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        topk2 = max(1, int(N1 / self.k2))
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, dtype=x.dtype)
        index = torch.topk(attn, k=topk2, dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        out = out1 * self.attn1 + out2 * self.attn2

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        if need_upsample:
            x = F.interpolate(x, size=(original_h, original_w), mode='bilinear', align_corners=False)

        return x

