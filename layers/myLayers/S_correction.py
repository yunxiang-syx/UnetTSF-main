import numpy as np
import torch
from torch import nn


def S_Correction(x, x_pre, d_model):
    layer_norm = nn.LayerNorm(336).to("cuda:0")
    x = layer_norm(x[:, :, :x_pre.shape[2]])
    x_pre = layer_norm(x_pre[:, :, :])
    x_fft = torch.fft.rfft(x,dim=1,norm='ortho')
    x_pre_fft = torch.fft.rfft(x_pre, dim=1, norm='ortho')
    x_fft = x_fft * torch.conj(x_fft)
    x_pre_fft = x_pre_fft * torch.conj(x_pre_fft)
    x_ifft = torch.fft.irfft(x_fft, dim=1) #
    x_pre_ifft = torch.fft.irfft(x_pre_fft, dim=1)
    x_ifft = torch.clamp(x_ifft,min=0)
    x_pre_ifft = torch.clamp(x_pre_ifft,min=0)
    alpha = torch.sum(x_ifft*x_pre_ifft,dim=1,keepdim=True)/(torch.sum(x_pre_ifft*x_pre_ifft,dim=1,keepdim=True)+0.001)
    #alpha = (x_ifft * x_pre_ifft) / (x_pre_ifft * x_pre_ifft + 0.001)
    return torch.sqrt(alpha)


input_data = torch.tensor(np.random.rand(256, 7, 432).astype(np.float32))
pred_data = torch.tensor(np.random.rand(256, 7, 336).astype(np.float32))
pred_data = S_Correction(input_data, pred_data, 336) * pred_data
print(pred_data,pred_data.shape[1])