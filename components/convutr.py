import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class DecomposedLargeKernel(nn.Module):
    """Decomposed large kernel convolution for efficiency.

    Instead of a single large kernel (e.g., 9x9), uses:
    - Multiple stacked small kernels (3x3) to achieve similar receptive field
    - Much lower computation: 3x(3x3) = 27 params vs 9x9 = 81 params

    Receptive field: k=3 -> 3, k=5 -> 5 (2x3x3), k=7 -> 7 (3x3x3), k=9 -> 9 (4x3x3)
    """
    def __init__(self, channels: int, target_kernel: int = 9):
        super().__init__()
                                                                      
        num_convs = (target_kernel - 1) // 2
        num_convs = max(1, num_convs)

        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            )
            if i < num_convs - 1:
                layers.append(nn.GELU())

        self.convs = nn.Sequential(*layers)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.convs(x)))


class ConvUtr(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, kernel: int = 9):
        super(ConvUtr, self).__init__()

        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(DecomposedLargeKernel(ch_in, target_kernel=kernel)),
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 2, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 2),
                    nn.Conv2d(ch_in * 2, ch_in, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for _ in range(depth)]
        )

        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

