import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)

        x_mean = x.mean(dim=1, keepdim=True)
        x_mean = x_mean.view(1, B, H, W)

        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )

        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        return x * att

