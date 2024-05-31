import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list): #[season_list[(1792, 128, 432),(1792, 128, 216),(1792, 128, 108),(1792, 128, 54)]]

        # mixing high->low
        out_high = season_list[0] #(1792, 128, 432)
        out_low = season_list[1] #(1792, 128, 216)
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list #[out_season_list[(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)]]


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list): #trend_list[(1792, 128, 432),(1792, 128, 216),(1792, 128, 108),(1792, 128, 54)]

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse() #(trend_list_reverse[(1792, 128, 54),(1792, 128, 108),(1792, 128, 216),(1792, 128, 432)])
        out_low = trend_list_reverse[0] # (1792, 128, 54)
        out_high = trend_list_reverse[1] # (1792, 128, 108)
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse() #[(1792, 54, 128),(1792, 108, 128),(1792, 216, 128),(1792, 432, 128)]
        return out_trend_list #[(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)]


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list): #x_list[(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)]
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T) #[(432, 216, 108, 54)]

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1)) #[season_list[(1792, 128, 432),(1792, 128, 216),(1792, 128, 108),(1792, 128, 54)]]
            trend_list.append(trend.permute(0, 2, 1)) #[trend_list[(1792, 128, 432),(1792, 128, 216),(1792, 128, 108),(1792, 128, 54)]]

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list) #(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list) #(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list #(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        if self.configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                       kernel_size=3, padding=padding,
                                       stride=self.configs.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        else:
            raise ValueError('Downsampling method is error,only supporting the max, avg, conv1D')

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list): ##x_list[(1792, 432, 1),(1792, 216, 1),(1792, 108, 1),(1792, 54, 1)]
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1) # x_enc(256, 7, 432)

        x_enc_ori = x_enc #(256, 7, 432)
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1)) #(256, 432, 7)
        x_mark_sampling_list.append(x_mark_enc) # [None]

        for i in range(self.configs.down_sampling_layers): # 3
            x_enc_sampling = self.down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list #[(256, 432, 7)，(256, 216, 7)，(256, 108, 7)，(256, 54, 7)]
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc #[(256, 432, 7)，(256, 216, 7)，(256, 108, 7)，(256, 54, 7)]   None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec): # x_enc: (256, 432, 7)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        # x_enc：[(256, 432, 7)，(256, 216, 7)，(256, 108, 7)，(256, 54, 7)]   None

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x) #x_list[(1792, 432, 1),(1792, 216, 1),(1792, 108, 1),(1792, 54, 1)]

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list) #tuple(x_list[(1792, 432, 1),(1792, 216, 1),(1792, 108, 1),(1792, 54, 1)]  None)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out) #[(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)]

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
            #enc_out_list[(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)]
        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list) #[(256, 336, 7),(256, 336, 7),(256, 336, 7),(256, 336, 7)]

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1) # (256, 336, 7)
        dec_out = self.normalize_layers[0](dec_out, 'denorm') # (256, 336, 7)
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        # B: 256
        # enc_out_list[(1792, 432, 128),(1792, 216, 128),(1792, 108, 128),(1792, 54, 128)]
        # x_list: tuple(x_list[(1792, 432, 1),(1792, 216, 1),(1792, 108, 1),(1792, 54, 1)]  None)
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0] #x_list[(1792, 432, 1),(1792, 216, 1),(1792, 108, 1),(1792, 54, 1)]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension #(1792, 336,128)
                dec_out = self.projection_layer(dec_out) #(1792, 336, 1)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous() #(256, 336, 7)
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list #[(256, 336, 7),(256, 336, 7),(256, 336, 7),(256, 336, 7)]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec) # x_enc：(256, 432, 7)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')


class Configs:
    def __init__(self):
        self.seq_len = 432 #输入序列长度
        self.enc_in = 7 # 输入通道数
        self.stage_num = 3
        self.stage_pool_kernel = 3
        self.stage_pool_padding = 0
        self.stage_pool_stride = 2
        self.pred_len = 336
        self.label_len = 0
        self.task_name = "long_term_forecast"
        self.enc_in = 7  # 输入通道数
        self.c_in = 7  # 输入通道数
        # self.decomposition = True
        self.d_model = 128
        self.c_out = 7
        self.d_ff = 32
        self.e_layers = 2
        self.down_sampling_layers = 3
        self.down_sampling_window = 2
        self.down_sampling_method = 'avg'
        self.learning_rate = 0.01
        self.channel_independence = 1
        self.moving_avg = 25
        self.dropout = 0.1
        self.top_k = 5
        self.use_norm = 1
        self.decomp_method = 'moving_avg'
        self.embed = 'timeF' #time features encoding, options:[timeF, fixed, learned]
        self.freq = 'h'
        self.features= 'M'

if __name__=='__main__':
    configs = Configs()
    past_series = torch.rand(256, configs.seq_len, 7)
    batch_y = torch.rand(256, 336, 7)
    model = Model(configs)
    pred_series = model(past_series, None, None, None)
    print(pred_series.shape) #torch.Size([256, 336, 7])