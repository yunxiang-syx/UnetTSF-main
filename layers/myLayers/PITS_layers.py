__all__ = ['moving_avg', 'series_decomp']           

import torch
from torch import nn
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x): #(256, 432, 7)
        # padding on the both ends of time series
        #参数 1 表示不复,第二个参数 (self.kernel_size - 1) // 2 表示沿着第二个维度（即时间步维度）复制 (self.kernel_size - 1) // 2 次
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) #(256, 12, 7)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  #(256, 12, 7)
        x = torch.cat([front, x, end], dim=1) #(256, 456, 7)
        x = self.avg(x.permute(0, 2, 1)) #(256, 7, 432)
        x = x.permute(0, 2, 1) #(256, 432, 7)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x): #(256, 432, 7)
        moving_mean = self.moving_avg(x) #(256, 432, 7)
        res = x - moving_mean #(256, 432, 7)
        return res, moving_mean
    
    