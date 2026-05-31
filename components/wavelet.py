from contextlib import nullcontext

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
                                                                            

class WTFD(nn.Module):

    def __init__(self, in_ch, out_ch, mode: str = "fusion"):
        super(WTFD, self).__init__()

        if mode not in {"low", "high", "fusion"}:
            raise ValueError(f"WTFD mode must be 'low', 'high' or 'fusion', got {mode}")
        self.mode = mode

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        orig_dtype = x.dtype
        needs_cast = orig_dtype in (torch.float16, torch.bfloat16)
        autocast_disabled = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if torch.is_autocast_enabled()
            else nullcontext()
        )
        with autocast_disabled:
            if needs_cast:
                x = x.to(torch.float32)
            yL, yH = self.wt(x)

            yH0 = yH[0]                                    
            y_HL = yH0[:, :, 0, :, :]                   
            y_LH = yH0[:, :, 1, :, :]
            y_HH = yH0[:, :, 2, :, :]
            yH_cat = torch.cat([y_HL, y_LH, y_HH], dim=1)                    
            yH_feat = self.conv_bn_relu(yH_cat)                            

            yL_out = self.outconv_bn_relu_L(yL)                                
            yH_out = self.outconv_bn_relu_H(yH_feat)                           

            if self.mode == "low":
                y = yL_out
            elif self.mode == "high":
                y = yH_out
            else:            
                y = self.alpha * yL_out + (1.0 - self.alpha) * yH_out
        if needs_cast:
            y = y.to(orig_dtype)

        return y

