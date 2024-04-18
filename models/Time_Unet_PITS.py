# __all__ = ['PatchTST_ours']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.myLayers.PITS_backbone import PITS_backbone
from layers.myLayers.PITS_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, 
                  verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        d_model = configs.d_model
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        c_in = configs.c_in
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        shared_embedding = configs.shared_embedding
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            
        else:
            self.model = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel](256, 432, 7)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x) #res_init(256, 432, 7), trend_init(256, 432, 7)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1) #(256, 7, 432) # x: [Batch, Channel, Input length]
            res = self.model_res(res_init) #(256, 7, 336)
            trend = self.model_trend(trend_init) #(256, 7, 336)
            x = res + trend #(256, 7, 336)
            x = x.permute(0,2,1) #(256, 336, 7)   # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x

class Configs:
    def __init__(self):
        self.seq_len = 432 #输入序列长度
        self.individual = True # 是否独立处理每个频道
        self.c_in = 7 # 输入通道数
        self.patch_len = 24
        self.stride = 12
        self.pred_len = 336
        self.shared_embedding = False
        self.decomposition = True
        self.d_model = 128
        self.head_dropout = 0.5
        self.padding_patch = 'end'
        self.revin = True
        self.affine = True
        self.subtract_last = False
        self.kernel_size = 25

if __name__=='__main__':
    configs = Configs()
    past_series = torch.rand(256, configs.seq_len, 7)
    model = Model(configs)
    pred_series = model(past_series) #(256, 336, 7)
    print(pred_series.shape)