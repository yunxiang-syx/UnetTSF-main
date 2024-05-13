import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from FPN.PANet.PANetFPN import PANetFPN

# 自定义一个维度重排模块
class PermuteModule(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)

class Generator(nn.Module):
    def __init__(self, layers, noise_len, n_samples, alpha, stage_num, n_channels):
        super().__init__()
        self.noise_len = noise_len
        self.n_channels = n_channels
        # 创建一个维度重排模块的实例
        permute_module = PermuteModule()
        self.stage_num = stage_num

        self.model = nn.Sequential(
            *layers,
            nn.Linear(noise_len, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(512, noise_len)
        )

    def forward(self, x): #x (B,C,T)
        e_left = []

        output = self.model[0](x)  #(256, 7, 336)# 336
        e_left.append(output)
        output_maxpool = self.model[1](x) # (256, 7, 215)# 215

        output = self.model[2](output_maxpool) #167 (256, 7, 167)
        e_left.append(output)
        output_maxpool = self.model[3](output_maxpool) #107  (256, 7, 107)

        output = self.model[4](output_maxpool) #83 (256, 7, 83)
        e_left.append(output)  # [(256, 7, 336), (256, 7, 167), (256, 7, 83)]

        # output_maxpool = self.model[5](output_maxpool) 最后一层avgpool并没有用到

        e_last = e_left[self.stage_num - 1] #(256, 7, 83)

        for i in range(self.stage_num - 1):
            e_last = torch.cat((e_left[self.stage_num - i - 2], e_last), dim=2)  # (256,7,250)
            e_last = self.model[6][i](e_last)  # (256,7,167) 从第六层开始
            # 最终e_last是(256, 7, 336)
        output = nn.Linear(e_last.shape[2], self.noise_len).to("cuda:0")(e_last) #(256, 7, 432)

        # 使用 view 方法将张量重塑为形状为 (256, 336) 的张量
        reshaped_tensor = output.view(256, -1)
        output = nn.Linear(output.shape[1] * output.shape[2], self.noise_len).to("cuda:0")(reshaped_tensor)

        for i in range(7, len(self.model)):
            output = self.model[i](output)

        return output


class Discriminator(nn.Module):
    def __init__(self, layers, n_samples, alpha, stage_num):
        super().__init__()

        # 创建一个维度重排模块的实例
        permute_module = PermuteModule()
        self.n_samples = n_samples
        self.stage_num = stage_num

        self.model = nn.Sequential(
        #    *layers,
            nn.Linear(n_samples, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        # e_left = []
        # x = x.permute(0, 2, 1)
        # output = self.model[0](x)  # 336
        # e_left.append(output)
        # output_maxpool = self.model[1](x)  # 215
        #
        # output = self.model[2](output_maxpool)  # 167
        # e_left.append(output)
        # output_maxpool = self.model[3](output_maxpool)  # 107
        #
        # output = self.model[4](output_maxpool)  # 83
        # e_left.append(output)
        # # output_maxpool = self.model[5](output_maxpool) 最后一层avgpool并没有用到
        #
        # e_last = e_left[self.stage_num - 1]
        #
        # for i in range(self.stage_num - 1):
        #     e_last = torch.cat((e_left[self.stage_num - i - 2], e_last), dim=2)  # (256,7,250)
        #     e_last = self.model[6][i](e_last)  # (256,7,167) 从第六层开始
        #     # 最终e_last是(256, 7, 336)
        #
        # # 使用 view 方法将张量重塑为形状为 (256, 336) 的张量
        # reshaped_tensor = e_last.view(256, -1)
        # e_last = nn.Linear(e_last.shape[1] * e_last.shape[2], n_samples).to("cuda:0")(reshaped_tensor)
        #
        # for i in range(7, len(self.model)):
        #     e_last = self.model[i](e_last)

       # output = nn.Linear(e_last.shape[2], self.n_samples).to("cuda:0")(e_last)

        return output

class Gan_model(nn.Module):
    def __init__(self, args, shared_model):
        super().__init__()
        self.args = args

        # Initialize Generator, Discriminator, and Central Discriminator
        self.noise_len = self.args.seq_len  # Length of noise vector
        self.n_samples = self.args.seq_len  # Number of output samples
        self.alpha = 0.1  # Leaky ReLU slope
        self.n_channels = self.args.enc_in  # Number of channels
        self.gamma_value = 5.0
        self.loss_function = nn.BCELoss()
        self.stage_num = self.args.stage_num

        # 提取公共模型的层
        self.layers = []
        # 遍历 self.down_blocks 和 self.Maxpools 中的层，并将它们交替取出添加到 layers 列表中
        for down_block, maxpool in zip(shared_model.down_blocks, shared_model.Maxpools):
            self.layers.append(down_block)
            self.layers.append(maxpool)

        self.layers.append(shared_model.up_blocks)

     #   self.generator = Generator(self.layers, self.noise_len, self.n_samples, self.alpha)
    #    self.discriminator = Discriminator(self.layers, self.n_samples, self.alpha)
        self.discriminators = {}
        for i in range(self.n_channels):
            self.discriminators[i] = Discriminator(self.layers, self.n_samples, self.alpha, self.stage_num).apply(self.initialize_weights)
        for i in range(self.n_channels):
            self.discriminators[i] = self.discriminators[i].to("cuda:0")
            ##

        self.generators = {}
        for i in range(self.n_channels):
            self.generators[i] = Generator(self.layers, self.noise_len, self.n_samples, self.alpha, self.stage_num, self.n_channels).apply(self.initialize_weights)
            self.generators[i].to("cuda:0")

        self.gamma = [self.gamma_value] * self.args.train_epochs

        # Define Optimizers
        self.optimizers_D = {}
        for i in range(self.n_channels):
            self.optimizers_D[i] = torch.optim.Adam(self.discriminators[i].parameters(), lr=0.001, betas=[0.5, 0.9])
        self.optimizers_G = {}
        for i in range(self.n_channels):
            self.optimizers_G[i] = torch.optim.Adam(self.generators[i].parameters(), lr=0.001, betas=[0.5, 0.9])

        # central discriminator
        self.central_discriminator =  Discriminator(self.layers, self.n_samples * self.n_channels, self.alpha, self.stage_num)
        self.central_discriminator = self.central_discriminator.apply(self.initialize_weights)
        self.central_discriminator.to("cuda:0")
        self.optimizer_central_discriminator = torch.optim.Adam(self.central_discriminator.parameters(),
                                                           lr=0.0001, betas=[0.5, 0.9])

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, batch_x, epoch = 1):
        signal_group = {}
        for i in range(self.n_channels):
            signal_group[i] = batch_x[:, :, i].to("cuda:0")
        # Generate noise
        # noise = torch.randn(batch_y.shape[0], n_channels, noise_len)
        shared_noise = torch.randn((batch_x.shape[0], self.n_channels, self.noise_len)).float().to("cuda:0")

        # Generate fake data
        # generated_data = generator(noise).to("cuda:0")
        generated_samples = {}
        for i in range(self.n_channels):
            generated_samples[i] = self.generators[i](shared_noise).to("cuda:0").float()
            # generated_samples[i] = outputs[:, :, i].float()

        generated_samples_labels = torch.zeros((batch_x.shape[0], 1)).to("cuda:0").float()
        real_samples_labels = torch.ones((batch_x.shape[0], 1)).to("cuda:0").float()
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Data for training the discriminators
        all_samples_group = {}
        for i in range(self.n_channels):
            all_samples_group[i] = torch.cat(
                (signal_group[i], generated_samples[i])
            )

        # Training the discriminators
        outputs_D = {}
        loss_D = {}
        for i in range(self.n_channels):
            self.optimizers_D[i].zero_grad()
            outputs_D[i] = self.discriminators[i](all_samples_group[i].float())
            loss_D[i] = self.loss_function(outputs_D[i], all_samples_labels)
            loss_D[i].backward(retain_graph=True)
            self.optimizers_D[i].step()

        # CD
        temp_generated = generated_samples[0]
        for i in range(1, self.n_channels):
            temp_generated = torch.hstack((temp_generated, generated_samples[i]))
        group_generated = temp_generated

        temp_real = signal_group[0]
        for i in range(1, self.n_channels):
            temp_real = torch.hstack((temp_real, signal_group[i]))
        group_real = temp_real

        all_samples_central = torch.cat((group_generated, group_real))
        all_samples_labels_central = torch.cat(
            (torch.zeros((batch_x.shape[0], 1)).to("cuda:0").float(),
             torch.ones((batch_x.shape[0], 1)).to("cuda:0").float())
        )

        # Training the central discriminator
        self.optimizer_central_discriminator.zero_grad()
        output_central_discriminator = self.central_discriminator(all_samples_central.float())
        loss_central_discriminator = self.loss_function(
            output_central_discriminator, all_samples_labels_central)
        loss_central_discriminator = loss_central_discriminator.clone().detach()
        loss_central_discriminator.requires_grad = True
        loss_central_discriminator.backward(retain_graph=True)
        self.optimizer_central_discriminator.step()

        # Training the generators
        outputs_G = {}
        loss_G_local = {}
        loss_G = {}
        for i in range(self.n_channels):
            self.optimizers_G[i].zero_grad()
            outputs_G[i] = self.discriminators[i](generated_samples[i])
            loss_G_local[i] = self.loss_function(outputs_G[i], real_samples_labels)
            loss_G_local[i] = loss_G_local[i].clone().detach()
            loss_G_local[i].requires_grad = True
            all_samples_central_new = {}
            output_central_discriminator_new = {}
            loss_central_discriminator_new = {}

            generated_samples_new = {}
            for j in range(self.n_channels):
                generated_samples_new[j] = self.generators[j](shared_noise)

                if i == j:
                    generated_samples_new[j] = generated_samples_new[j].float()
                else:
                    generated_samples_new[j] = generated_samples_new[j].detach().float()
            temp_generated = generated_samples_new[0]
            for j in range(1, self.n_channels):
                temp_generated = torch.hstack((temp_generated, generated_samples_new[j]))
            all_generated_samples = temp_generated
            all_samples_central_new[i] = torch.cat((all_generated_samples, group_real))
            output_central_discriminator_new[i] = self.central_discriminator(all_samples_central_new[i].float())
            loss_central_discriminator_new[i] = self.loss_function(
                output_central_discriminator_new[i], all_samples_labels_central)

            loss_G[i] = loss_G_local[i] - self.gamma[epoch] * loss_central_discriminator_new[i]
            loss_G[i].backward(retain_graph=True)
            self.optimizers_G[i].step()

        return loss_central_discriminator

class Configs:
    def __init__(self):
        self.seq_len = 432 #输入序列长度
        self.individual = True # 是否独立处理每个频道
        self.enc_in = 7 # 输入通道数
        self.cut_freq = 50  # 截断频率，即考虑的最大频率
        self.stage_num = 3
        self.stage_pool_kernel = 3
        self.stage_pool_padding = 0
        self.stage_pool_stride = 2
        self.pred_len = 336
        self.train_epochs = 100

if __name__=='__main__':
    configs = Configs()
    past_series = torch.rand(256, configs.seq_len, 7)
    n_samples = configs.pred_len
    shared_model = PANetFPN(configs)
    gan_model = Gan_model(configs, shared_model)
    loss = gan_model(past_series)
    print(loss)

    # Sequential(
    #     (0): ModuleList(
    #     (0): block_model(
    #     (Linear_channel): ModuleList(
    #     (0): Linear(in_features=432, out_features=336, bias=True)
    # (1): Linear(in_features=432, out_features=336, bias=True)
    # (2): Linear(in_features=432, out_features=336, bias=True)
    # (3): Linear(in_features=432, out_features=336, bias=True)
    # (4): Linear(in_features=432, out_features=336, bias=True)
    # (5): Linear(in_features=432, out_features=336, bias=True)
    # (6): Linear(in_features=432, out_features=336, bias=True)
    # )
    # (ln): LayerNorm((336,), eps=1e-05, elementwise_affine=True)
    # (relu): ReLU(inplace=True)
    # )
    # (1): block_model(
    #     (Linear_channel): ModuleList(
    #     (0): Linear(in_features=215, out_features=167, bias=True)
    # (1): Linear(in_features=215, out_features=167, bias=True)
    # (2): Linear(in_features=215, out_features=167, bias=True)
    # (3): Linear(in_features=215, out_features=167, bias=True)
    # (4): Linear(in_features=215, out_features=167, bias=True)
    # (5): Linear(in_features=215, out_features=167, bias=True)
    # (6): Linear(in_features=215, out_features=167, bias=True)
    # )
    # (ln): LayerNorm((167,), eps=1e-05, elementwise_affine=True)
    # (relu): ReLU(inplace=True)
    # )
    # (2): block_model(
    #     (Linear_channel): ModuleList(
    #     (0): Linear(in_features=107, out_features=83, bias=True)
    # (1): Linear(in_features=107, out_features=83, bias=True)
    # (2): Linear(in_features=107, out_features=83, bias=True)
    # (3): Linear(in_features=107, out_features=83, bias=True)
    # (4): Linear(in_features=107, out_features=83, bias=True)
    # (5): Linear(in_features=107, out_features=83, bias=True)
    # (6): Linear(in_features=107, out_features=83, bias=True)
    # )
    # (ln): LayerNorm((83,), eps=1e-05, elementwise_affine=True)
    # (relu): ReLU(inplace=True)
    # )
    # )
    # (1): ModuleList(
    #     (0): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(0,))
    # (1): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(0,))
    # (2): AvgPool1d(kernel_size=(3,), stride=(2,), padding=(0,))
    # )
    # (2): ModuleList(
    #     (0): block_model(
    #     (Linear_channel): ModuleList(
    #     (0): Linear(in_features=250, out_features=167, bias=True)
    # (1): Linear(in_features=250, out_features=167, bias=True)
    # (2): Linear(in_features=250, out_features=167, bias=True)
    # (3): Linear(in_features=250, out_features=167, bias=True)
    # (4): Linear(in_features=250, out_features=167, bias=True)
    # (5): Linear(in_features=250, out_features=167, bias=True)
    # (6): Linear(in_features=250, out_features=167, bias=True)
    # )
    # (ln): LayerNorm((167,), eps=1e-05, elementwise_affine=True)
    # (relu): ReLU(inplace=True)
    # )
    # (1): block_model(
    #     (Linear_channel): ModuleList(
    #     (0): Linear(in_features=503, out_features=336, bias=True)
    # (1): Linear(in_features=503, out_features=336, bias=True)
    # (2): Linear(in_features=503, out_features=336, bias=True)
    # (3): Linear(in_features=503, out_features=336, bias=True)
    # (4): Linear(in_features=503, out_features=336, bias=True)
    # (5): Linear(in_features=503, out_features=336, bias=True)
    # (6): Linear(in_features=503, out_features=336, bias=True)
    # )
    # (ln): LayerNorm((336,), eps=1e-05, elementwise_affine=True)
    # (relu): ReLU(inplace=True)
    # )
    # )
    # (3): PermuteModule()
    # (4): Linear(in_features=432, out_features=256, bias=True)
    # (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (6): LeakyReLU(negative_slope=0.1)
    # (7): Dropout(p=0.3, inplace=False)
    # (8): Linear(in_features=256, out_features=512, bias=True)
    # (9): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (10): LeakyReLU(negative_slope=0.1)
    # (11): Dropout(p=0.3, inplace=False)
    # (12): Linear(in_features=512, out_features=336, bias=True)
    # )