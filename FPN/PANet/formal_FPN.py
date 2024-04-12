import torch
import torch.nn as nn


class FPNPyramid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNPyramid, self).__init__()
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Downsample layers
        for i in range(4):
            if i == 0:
                self.downsample_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))

        # Upsample layers
        for i in range(3):
            self.upsample_layers.append(
                nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        # Last upsample layer
        self.last_upsample_layer = nn.ConvTranspose1d(out_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x): #(256, 7, 432)
        # x = x.permute(0, 2, 1) #(256, 7, 432)
        # Downsample
        downsampled_features = []
        for downsample_layer in self.downsample_layers:
            x = downsample_layer(x)
            downsampled_features.append(x)
        #downsampled_features={(256, 336, 216), (256, 336, 108), (256, 336, 54), (256, 336, 27)}
        # Upsample and merge
        output = downsampled_features[-1] #(256, 336, 27) Start with the last downsampled feature
        for i, upsample_layer in enumerate(self.upsample_layers):
           # output += upsample_layer(downsampled_features[-(i + 2)])  # Add the corresponding downsampled feature
           output = upsample_layer(output) # (256, 336, 54)
           output = output + downsampled_features[-(i + 2)] # (256, 336, 54)

        # Last upsample layer
        output = self.last_upsample_layer(output) # (256, 336, 216)  ->  (256, 7 ,432)

        device = output.device  # 将输出张量移动到与输入张量相同的设备上
        linear_layer = nn.Linear(output.size(2), self.out_channels).to(device)  # 创建新的Linear层，并将其移动到相同的设备上
        output = linear_layer(output)  # 使用新的Linear层进行操作 #(256, 7, 336)

        output = output.permute(0, 2, 1)
      #  output = output[:, :, -336:].transpose(1, 2) #[256, 336, 7] 不用在这儿转，在外边有代码转成pred_len
        return output

if __name__ == "__main__":

    # Example usage
    B, T, C = 256, 96, 7  # Batch size, sequence length, number of channels
    in_channels = C  # Number of input channels
    out_channels = 336  # Number of output channels
    Ni = torch.randn(B, C, T)  # Input feature sequence Ni

    # Create CustomPyramid module
    custom_pyramid = FPNPyramid(in_channels, out_channels)

    # Forward pass
    output_feature = custom_pyramid(Ni) # Ni: (256, 7, 432)

    # Check the output shape
    print(output_feature.shape)  # Output shape should be torch.Size([256, 336, 7])