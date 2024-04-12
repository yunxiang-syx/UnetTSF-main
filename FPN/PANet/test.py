import torch
import torch.nn as nn


class CustomFPN(nn.Module):
    def __init__(self, in_channels, out_channels): # 7    336
        super(CustomFPN, self).__init__()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.prev_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
   #卷积的in_channels看的是(A,B,C)里的B
    def forward(self, x, prev_feature):
        # x:(512, 7, 432)   prev_feature:(512, 7, 216)
        downsampled_feature = self.downsample(x) #(512, 336, 216) (B, Out_channels, 216)

        prev_feature = self.prev_conv(prev_feature) # (512, 336, 216)

        # Element-wise addition
        combined_feature = downsampled_feature + prev_feature # (512, 336, 216)

        # Apply convolution
        output_feature = self.conv(combined_feature).transpose(1, 2) #torch.Size([512, 216, 336]) (B,Middle,out_channels)

        return output_feature


# Example usage
B, T, C = 512, 432, 7  # Batch size, sequence length, number of channels
in_channels = C  # Number of input channels
out_channels = 336  # Number of output channels
Ni = torch.randn(B, C, T)  # Input feature sequence Ni
Pi_1 = torch.randn(B, C, T // 2)  # Previous feature sequence Pi+1

# Create CustomFPN module
custom_fpn = CustomFPN(in_channels, out_channels)

# Forward pass
output_feature = custom_fpn(Ni, Pi_1)

# Check the output shape
print(output_feature.shape)  # Output shape should be torch.Size([512, 216, 336])