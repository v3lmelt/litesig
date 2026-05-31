import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState


def complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape

    fan_in = input_chs

    fan_out = output_chs

    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState()

    kernel_shape = weights_real.shape

    modulus = rng.rayleigh(scale=s, size=kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_real = modulus * np.cos(phase)

    weight_imag = modulus * np.sin(phase)

    weights_real.data = torch.Tensor(weight_real)

    weights_imag.data = torch.Tensor(weight_imag)

class FourierStrokeCalibration(nn.Module):

    def __init__(self, input_chs: int, output_chs: int, num_rows: int = 224, num_cols: int = 224, stride=1, init='he'):
        super().__init__()

        self.input_chs = input_chs
        self.stride = stride
        self.num_rows = num_rows
        self.num_cols = num_cols

        freq_h = num_rows
        freq_w = num_cols // 2 + 1

        self.weights_real = nn.Parameter(
            torch.Tensor(1, input_chs, freq_h, freq_w)
        )

        self.weights_imag = nn.Parameter(
            torch.Tensor(1, input_chs, freq_h, freq_w)
        )

        complexinit(self.weights_real, self.weights_imag, init)

        self.norm = nn.BatchNorm2d(input_chs)

    def forward(self, x):
        _, _, H, W = x.shape

        if H != self.num_rows or W != self.num_cols:
            pass

        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        x_freq = torch.fft.rfftn(x, dim=(-2, -1), norm=None)


        freq_h, freq_w = x_freq.shape[-2], x_freq.shape[-1]

        x_real = x_freq.real
        x_imag = x_freq.imag

        weights_real, weights_imag = self._get_spectral_weights(freq_h, freq_w)

        y_real = torch.mul(x_real, weights_real) - torch.mul(x_imag, weights_imag)

        y_imag = torch.mul(x_real, weights_imag) + torch.mul(x_imag, weights_real)

        out = torch.fft.irfftn(
            torch.complex(y_real, y_imag),
            s=(H, W),
            dim=(-2, -1),
            norm=None
        )

        if self.stride == 2:
            out = out[..., ::2, ::2]

        out = self.norm(out)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        return out

    def _get_spectral_weights(self, target_h: int, target_w: int):
        ref_h, ref_w = self.weights_real.shape[2], self.weights_real.shape[3]

        if target_h == ref_h and target_w == ref_w:
            return self.weights_real, self.weights_imag

        device = self.weights_real.device
        dtype = self.weights_real.dtype
        B, C = 1, self.weights_real.shape[1]

        weights_real = torch.ones(B, C, target_h, target_w, device=device, dtype=dtype)
        weights_imag = torch.zeros(B, C, target_h, target_w, device=device, dtype=dtype)

        copy_h = min(target_h, ref_h)
        copy_w = min(target_w, ref_w)

        weights_real[:, :, :copy_h, :copy_w] = self.weights_real[:, :, :copy_h, :copy_w]
        weights_imag[:, :, :copy_h, :copy_w] = self.weights_imag[:, :, :copy_h, :copy_w]

        return weights_real, weights_imag

