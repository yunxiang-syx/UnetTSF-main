import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN



# block_model模块根据individual参数的取值决定是否为每个通道单独创建线性层，然后在forward方法中根据不同情况进行线性变换，最终输出经过线性变换后的结果
class block_model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, input_channels, input_len, out_len, individual):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual

        if self.individual:
            self.Linear_channel = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
        else:
            self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [Batch, Input length, Channel] 不对吧
        # x: [Batch, Channel, Input Length] (256, 7, 432) 即(256,7,input_len(越来越小，会变))
        if self.individual:
            # output(256,7,336) 即(256,7,out_len(越来越小，会变))
            output = torch.zeros([x.size(0), x.size(1), self.out_len], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, i, :] = self.Linear_channel[i](x[:, i, :])  # x(256, 7, 432) -> output(256,7,336)的过程
        else:
            output = self.Linear_channel(x)
        # output = self.ln(output) #   #output(256,7,336)
        # output = self.relu(output)
        return output  # [Batch, Channel, Output length]


class PANetFPN(nn.Module):
    def __init__(self, configs):
        super(PANetFPN, self).__init__()

        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        # 下采样设定
        self.stage_num = configs.stage_num
        self.stage_pool_kernel = configs.stage_pool_kernel
        self.stage_pool_stride = configs.stage_pool_stride
        self.stage_pool_padding = configs.stage_pool_padding

        len_in = self.input_len
        len_out = self.out_len
        down_in = [len_in] #([432, 215, 107])
        down_out = [len_out] #([336, 167, 83])
        i = 0
        while i < self.stage_num - 1:
            linear_in = int(
                (len_in + 2 * self.stage_pool_padding - self.stage_pool_kernel) / self.stage_pool_stride + 1)
            linear_out = int(
                (len_out + 2 * self.stage_pool_padding - self.stage_pool_kernel) / self.stage_pool_stride + 1)
            down_in.append(linear_in)
            down_out.append(linear_out)
            len_in = linear_in
            len_out = linear_out
            i = i + 1

        # 最大池化层
        self.Maxpools = nn.ModuleList()
        # 左边特征提取层
        self.down_blocks = nn.ModuleList()
        for in_len, out_len in zip(down_in, down_out):
            self.down_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))
            self.Maxpools.append(nn.AvgPool1d(kernel_size=self.stage_pool_kernel, stride=self.stage_pool_stride,
                                              padding=self.stage_pool_padding))

        # 特征融合层
        self.up_blocks = nn.ModuleList()
        len_down_out = len(down_out)
        for i in range(len_down_out - 1):
            print(len_down_out, len_down_out - i - 1, len_down_out - i - 2)
            in_len = down_out[len_down_out - i - 1] + down_out[len_down_out - i - 2]
            out_len = down_out[len_down_out - i - 2]
            self.up_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))


        # 最右边特征融合
        self.add_blocks = nn.ModuleList()
        for i in range(len_down_out - 1):
            # in_len = down_out[i]
            # out_len = down_out[i+1]
            # self.add_blocks.append(nn.Linear(in_len, out_len))
            self.add_blocks.append(nn.AvgPool1d(kernel_size=self.stage_pool_kernel, stride=self.stage_pool_stride,
                         padding=self.stage_pool_padding))

        #最后线性层融合特征
        self.last_linear = nn.Linear(sum(down_out),self.out_len)
        # self.linear_out = nn.Linear(self.out_len * 2, self.out_len)

    def forward(self, x):  # x:(256,7,432)

        e_left = []
        i = 0
        for down_block in self.down_blocks:
            e_left.append(down_block(x))
            x = self.Maxpools[i](x)
            i = i + 1
        #e_left{(256, 7, 336), (256, 7, 167), (256, 7, 83)}

        e_middle = [e_left[self.stage_num - 1]]
        e_last = e_left[self.stage_num - 1]  # (256,7,83)

        for i in range(self.stage_num - 1):
            e_last = torch.cat((e_left[self.stage_num - i - 2], e_last), dim=2)  # (256,7,250)
            e_last = self.up_blocks[i](e_last)  # (256,7,167)
            e_middle.append(e_last)
        #e_middle{(256, 7, 83), (256, 7, 167), (256, 7, 336)}

        e_right=[e_middle[self.stage_num - 1]]
        e_last = e_middle[self.stage_num - 1] #(256, 7, 336)
        for i in range(self.stage_num - 1):
            e_last = self.add_blocks[i](e_last) #(266, 7, 167)
           # e_last = self.down_blocks[i](e_last)
            e_last = e_last + e_middle[self.stage_num - i - 2] ##(266, 7, 167)
           # e_last = torch.cat((e_left[self.stage_num - i - 2], e_last), dim=2)

            e_right.append(e_last)
        #e_right{(256, 7, 336), (256, 7 ,167), (256, 7, 83)}

        # 遍历列表中的每个张量并拼接它们的最后一维
        for i in range(len(e_right)-1):
            e_right[i+1] = torch.cat((e_right[i], e_right[i+1]), dim=2)
            concatenated_tensor = e_right[i+1]
        #concatenated_tensor(256, 7, 586)

        e_last = self.last_linear(concatenated_tensor)
        e_last = e_last.permute(0, 2, 1)  # 最后(256,336,7)

        return e_last


class Configs:
    def __init__(self):
        self.seq_len = 432 #输入序列长度
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
    model = PANetFPN(configs)
    pred_series = model(past_series.permute(0,2,1))
    print(pred_series.shape)
#
# #ModuleList(
#   (0): block_model(
#     (Linear_channel): Linear(in_features=432, out_features=336, bias=True)
#     (ln): LayerNorm((336,), eps=1e-05, elementwise_affine=True)
#     (relu): ReLU(inplace=True)
#   )
#   (1): block_model(
#     (Linear_channel): Linear(in_features=215, out_features=167, bias=True)
#     (ln): LayerNorm((167,), eps=1e-05, elementwise_affine=True)
#     (relu): ReLU(inplace=True)
#   )
#   (2): block_model(
#     (Linear_channel): Linear(in_features=107, out_features=83, bias=True)
#     (ln): LayerNorm((83,), eps=1e-05, elementwise_affine=True)
#     (relu): ReLU(inplace=True)
#   )
# )

# ModuleList(
#   (0): block_model(
#     (Linear_channel): Linear(in_features=250, out_features=167, bias=True)
#     (ln): LayerNorm((167,), eps=1e-05, elementwise_affine=True)
#     (relu): ReLU(inplace=True)
#   )
#   (1): block_model(
#     (Linear_channel): Linear(in_features=503, out_features=336, bias=True)
#     (ln): LayerNorm((336,), eps=1e-05, elementwise_affine=True)
#     (relu): ReLU(inplace=True)
#   )
# )
#
# ModuleList(
#   (0): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(0,))
#   (1): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(0,))
#   (2): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(0,))
# )