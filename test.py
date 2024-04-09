import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedFPN_TimeSeries(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(CentralizedFPN_TimeSeries, self).__init__()
        self.fc_c5 = nn.Linear(input_size, hidden_size)
        self.fc_c4 = nn.Linear(input_size, hidden_size)
        self.fc_c3 = nn.Linear(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, c5, c4, c3):
        h_c5 = self.fc_c5(c5)
        h_c4 = self.fc_c4(c4) + F.interpolate(h_c5, size=c4.size()[1:], mode='nearest')
        h_c3 = self.fc_c3(c3) + F.interpolate(h_c4, size=c3.size()[1:], mode='nearest')

        output = self.fc(h_c3)

        return output


# 示例输入
c5 = torch.randn(1, 20, 1)  # 假设时间序列长度为20
c4 = torch.randn(1, 40, 1)  # 假设时间序列长度为40
c3 = torch.randn(1, 80, 1)  # 假设时间序列长度为80

model = CentralizedFPN_TimeSeries()
outputs = model(c5, c4, c3)

print(outputs.shape)
