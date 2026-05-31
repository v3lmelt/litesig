import torch
import torch.nn as nn


class StrokeDirectionalAttention(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super().__init__()

        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()

        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        attention_mean = identity * x_w_mean * x_h_mean

        attention_max = identity * x_w_max * x_h_max

        return attention_mean + attention_max

