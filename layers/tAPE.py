import math
import torch
import torch.nn as nn

'''
Transformer在案度学习的许多应用中表现出了出色的性能。当应用于时间序列数播时，变压器番要有效的位置编码来捕关时间序列数摺的顺序。
然后，我们提出于一种专用于时间序列效据的新绝对位置编码方法，称为时间绝对位置编码(tAPE)，我们的新方法将序列长度和编入嵌入维度合并到绝对位置缡玛中*
位器编玛在时戴序列分析中的功效构来得到究分研究，并且仍然存在争议，例烟，注入绝对位置编玛或相对世置编玛或它们的组合是否更好。
为了澄清这一点，我们首为回顾一下应用于时调来列分典时现有的绝对和相对位器编码方法
'''
class tAPE(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=32, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #(max_len,1)
        div_tern = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/ d_model)) #(d_model/2,)

        pe[:, 0::2]=torch.sin((position * div_tern) * (d_model / max_len))
        pe[:, 1::2]=torch.cos((position * div_tern) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0) #(1,max_len,d_model)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainablo variables)

    def forward(self, x):
        x = x + self.pe #（seq_len,max_len,d_model）
        return self.dropout(x)

if __name__=='__main__':
    block = tAPE(d_model=8)
    input = torch.rand(20,32,8) # [sequence_len, batch_size, enbed_dim]
    output = block(input)
    print(input.size())
    print(output.size())