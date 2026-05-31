import torch
import torch.nn as nn


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, ratio=2):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)

        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)

        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)

        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)

        out = self.gamma * weights + x
        return out


class CSAM(nn.Module):

    def __init__(self, in_channels, ratio=2):
        super(CSAM, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels, ratio)
        self.cab = ChannelAttentionBlock(in_channels)

    def forward(self, x):
        sab_out = self.sab(x)
        cab_out = self.cab(x)
        out = sab_out + cab_out
        return out

