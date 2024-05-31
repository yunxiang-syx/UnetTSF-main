import torch
import torch.nn as nn

from FPN.BiFPN import BiFPN
from FPN.PANet.PANetFPN import PANetFPN
from FPN.PANet.formal_FPN import FPNPyramid
from layers.RevIN import RevIN
from layers.myLayers.PITS_backbone import PITS_backbone
from layers.myLayers.PITS_layers import series_decomp
from layers.myLayers.RecursiveFPN import RecursiveFPN


class Model(nn.Module):
    def __init__(self, configs, verbose:bool=False, **kwargs):
        super(Model, self).__init__()
        # self.fpn_pyramid = FPNPyramid(configs.enc_in, configs.pred_len)
        # load parameters
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.pred_len = configs.pred_len
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
        self.fpn_pyramid = PANetFPN(configs)
      #  self.fpn_pyramid = RecursiveFPN(configs)
     #   self.bifpn_pyramid = BiFPN(configs.enc_in, feature_size=configs.bifpn_features, num_layers=configs.bifpn_numlayers)
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.decomp_module = series_decomp(kernel_size = 25)
        self.model_trend = PITS_backbone(c_in=c_in,
                                         context_window=context_window, target_window=context_window,
                                         patch_len=patch_len, stride=stride,
                                         d_model=d_model,
                                         shared_embedding=shared_embedding,
                                         head_dropout=head_dropout,
                                         padding_patch=padding_patch,
                                         individual=individual, revin=revin, affine=affine,
                                         subtract_last=subtract_last, verbose=verbose, **kwargs)
        self.model_res = PITS_backbone(c_in=c_in,
                                       context_window=context_window, target_window=context_window, patch_len=patch_len,
                                       stride=stride,
                                       d_model=d_model,
                                       shared_embedding=shared_embedding,
                                       head_dropout=head_dropout,
                                       padding_patch=padding_patch,
                                       individual=individual, revin=revin, affine=affine,
                                       subtract_last=subtract_last, verbose=verbose, **kwargs)
    def forward(self, x):  # x:(256,432,7)  进金字塔前是（B,C,T）出金字塔后是（B,pred_len,C）
        x = self.revin_layer(x, 'norm')
        # res_init, trend_init = self.decomp_module(x)  # res_init(256, 432, 7), trend_init(256, 432, 7)
        # res_init, trend_init = res_init.permute(0, 2, 1),  trend_init.permute(0, 2, 1)  # (256, 7, 432) #
        # res_fpn = self.fpn_pyramid(res_init)
        # trend_fpn = self.fpn_pyramid(trend_init)
        # e_last = res_fpn + trend_fpn
        x = x.permute(0, 2, 1)  # x1 (256,7,432)
   #     output = self.fpn_pyramid(x)  # (256,336,7)
        output = self.fpn_pyramid(x)
   #      x = x.permute(0, 2, 1)  # x1 (256,7,432)
   #      output = self.bifpn_pyramid(x, self.pred_len)
   #      output = output.permute(0, 2, 1)
     #   output = self.fpn_pyramid(res) + self.fpn_pyramid(trend)
        e_last = self.revin_layer(output, 'denorm')  # (256,432,7)
       # e_last = self.revin_layer(e_last, 'denorm')
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
        self.seq_len = 432  # 输入序列长度
        self.individual = True  # 是否独立处理每个频道
        self.c_in = 7  # 输入通道数
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
        self.bifpn_features = 128
        self.bifpn_numlayers = 3

if __name__=='__main__':
    configs = Configs()
    past_series = torch.rand(256, configs.seq_len, 7)
    model = Model(configs)
    pred_series = model(past_series)
    print(pred_series.shape)