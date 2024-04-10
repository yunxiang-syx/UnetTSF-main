#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from CFPnet import darknet
from CFPnet import evc_blocks


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()

        self.backbone = darknet.Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return evc_blocks.BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs

if __name__ == "__main__":
    model = YOLOFPN()
   # print(model)
    # 定义张量的维度大小
    B = 256  # 批量大小
    C = 7  # 时间步数
    T = 50  # 高度
    N = 66  # 宽度

    # 生成随机的四维张量
    random_tensor = np.random.rand(B, C, T, N)
    # 将 NumPy 数组转换为 PyTorch 张量
    input_tensor = torch.tensor( random_tensor)
    output = model(input_tensor.double())
    print(output.shape)