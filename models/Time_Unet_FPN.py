import torch
import torch.nn as nn

from FPN.PANet.PANetFPN import PANetFPN
from FPN.PANet.formal_FPN import FPNPyramid
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
     #   self.fpn_pyramid = FPNPyramid(configs.enc_in, configs.pred_len)
        self.fpn_pyramid = PANetFPN(configs)
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

    def forward(self, x):  # x:(256,432,7)
        x = self.revin_layer(x, 'norm')
        x1 = x.permute(0, 2, 1)  # x1 (256,7,432)
        output = self.fpn_pyramid(x1) # (256,336,7)
        e_last = self.revin_layer(output, 'denorm')  # (256,432,7)
        return e_last


class Configs:
    def __init__(self):
        self.seq_len = 96 #输入序列长度
        self.individual = False # 是否独立处理每个频道
        self.enc_in = 7 # 输入通道数
        self.cut_freq = 50  # 截断频率，即考虑的最大频率
        self.stage_num = 3
        self.stage_pool_kernel = 3
        self.stage_pool_padding = 0
        self.stage_pool_stride = 2
        self.pred_len = 336

if __name__=='__main__':
    configs = Configs()
    past_series = torch.rand(256, configs.seq_len, 7)
    model = Model(configs)
    pred_series = model(past_series)
    print(pred_series.shape)