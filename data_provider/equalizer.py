import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Equalizer(nn.Module):
    def __init__(self, configs):
        super(Equalizer, self).__init__()
     #   self.linear = nn.Linear(configs.seq_len //2 +1, configs.pred_len//2 + 1)
        # select prior
        if configs.prior == 'mag' or configs.prior == 'slc':
            c_in = configs.enc_in
        elif configs.prior == 'self':
            c_in = configs.enc_in * 2
        else:
            c_in = 1

        # select equalizer
        self.equalizer = configs.equalizer
        if configs.equalizer == 'conv':
            self.eq = nn.Conv1d(
                in_channels = c_in,
                out_channels = 1,
                kernel_size = configs.eq_kernel_size,
                padding = configs.eq_kernel_size//2,
                bias = False)
        elif configs.equalizer == 'transformer':
            num_freq_steps = configs.seq_len//2 + 1
            encoder_layers = TransformerEncoderLayer(
                d_model = num_freq_steps,
                nhead = 1,
                dim_feedforward = 2*num_freq_steps,
                batch_first = True)
            self.eq = TransformerEncoder(encoder_layers, 2)

    def forward(self, I_f): #(256, 7, 217)
        if self.equalizer == 'conv':
            return F.sigmoid(self.eq(I_f))
        elif self.equalizer == 'transformer':
       #     I_f = self.linear(I_f)
            return F.sigmoid(torch.mean(self.eq(I_f), dim=1, keepdim=True))