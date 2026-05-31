import torch
import torch.nn as nn


def calculate_padding(kernel_size, padding=None, dilation=1):
    if dilation > 1:
        if isinstance(kernel_size, int):
            effective_kernel = dilation * (kernel_size - 1) + 1
        else:
            effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    else:
        effective_kernel = kernel_size

    if padding is None:
        if isinstance(effective_kernel, int):
            padding = effective_kernel // 2
        else:
            padding = [k // 2 for k in effective_kernel]
    return padding

class ConvolutionLayer(nn.Module):
    default_activation = nn.SiLU()

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            calculate_padding(kernel_size, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

        if activation is True:
            self.activation = self.default_activation
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.batch_norm(self.convolution(x)))

    def forward_fused(self, x):
        return self.activation(self.convolution(x))

class StrokeScalePerceptionConv(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation=True):
        super().__init__()

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            ConvolutionLayer(channels, channels, kernel_size=1, stride=1),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels),
            ConvolutionLayer(channels, channels, kernel_size=1, stride=1),
        )

        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels),
            ConvolutionLayer(channels, channels, kernel_size=1, stride=1),
        )

        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        out_3 = self.branch_3x3(x)
        out_5 = self.branch_5x5(x)
        out_7 = self.branch_7x7(x)

        weights = torch.softmax(self.scale_weights, dim=0)
        out = weights[0] * out_3 + weights[1] * out_5 + weights[2] * out_7
        return out

