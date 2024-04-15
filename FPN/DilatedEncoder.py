import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # 1x1 conv for channel adjustment
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # 3x3 conv for feature extraction
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Another 1x1 conv for channel adjustment after conv2
        self.conv3 = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)

        out += residual  # Residual connection
        out = F.relu(out)

        return out


class DilatedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks):
        super(DilatedEncoder, self).__init__()

        # Initial 1x1 conv layer for adjusting input channels
        self.conv_init = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # 3x3 conv layer for information extraction
        self.conv_main = nn.Conv1d(out_channels, in_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(in_channels, out_channels) for _ in range(num_residual_blocks)])

    def forward(self, x):
        out = F.relu(self.conv_init(x))
        out = F.relu(self.conv_main(out))

        # Applying residual blocks
        for block in self.res_blocks:
            out = block(out)

        return out


# 测试模块
input_channels = 7
output_channels = 2048
num_residual_blocks = 4

model = DilatedEncoder(input_channels, output_channels, num_residual_blocks)
input_tensor = torch.randn(256, input_channels, 432)  # 输入大小为(batch_size, channels, sequence_length)
output_tensor = model(input_tensor)
print("Output tensor shape:", output_tensor.shape)
