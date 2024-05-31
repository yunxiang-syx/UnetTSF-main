import torch
import torch.nn as nn
import torch.nn.functional as F

# SE Module
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool1d for time series
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.gap(x).view(b, c)
     #   y = self.fc(y).view(b, c, 1)
        return y
     #   return x * y.expand_as(x)


# Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.SE = SE_Block(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, self.expansion * outchannel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out = out + self.shortcut(x)
      #  out += self.shortcut(x)
        out = F.relu(out)
        return out


# Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.conv3 = nn.Conv1d(outchannel, self.expansion * outchannel,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * outchannel)
        self.SE = SE_Block(self.expansion * outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, self.expansion * outchannel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# SE-ResNet Architecture
class SE_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SE_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out


# SE-ResNet models
def SE_ResNet18():
    return SE_ResNet(BasicBlock, [2, 2, 2, 2])


def SE_ResNet34():
    return SE_ResNet(BasicBlock, [3, 4, 6, 3])


def SE_ResNet50():
    return SE_ResNet(Bottleneck, [3, 4, 6, 3])


def SE_ResNet101():
    return SE_ResNet(Bottleneck, [3, 4, 23, 3])


def SE_ResNet152():
    return SE_ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    # Example usage and summary
    # net = SE_ResNet50()
    # print(net)
    # input = torch.randn(1, 3, 224)
    # out = net(input)
    # print(out.shape)
    net = BasicBlock(7, 7)
    input = torch.randn(256, 7, 432)
    output = net(input)
    print(output.shape)
