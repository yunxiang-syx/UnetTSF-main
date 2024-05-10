from data_provider.augmentations import AugBoostDeep
from data_provider.data_factory import data_provider
from data_provider.equalizer import Equalizer
from exp.exp_basic import Exp_Basic
from layers.myLayers.Gan import Generator, Discriminator
from layers.tAPE import tAPE
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, Time_Unet,\
    Time_Unet_FITS,TimesNet,ModernTCN_Unet,Time_Unet_ModernBlock, Time_Unet_FPN,Time_Unet_PITS
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from data_provider.data_aug import noise_injection, amplitude_perturbation, clip_and_scale

import os
import time
import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile as thopprofile

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # 引入SimPSI
        self.eq = Equalizer(args).to(self.device)

        #引入SimPSI
        self.equalizer_optimizer = torch.optim.Adam(self.eq.parameters(), lr=3e-4,
                                               betas=(0.9, 0.99), weight_decay=3e-4)
        self.augboost = AugBoostDeep(aug_list=[], prior=args.prior)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'Time_Unet':Time_Unet,
            'Time_Unet_FITS': Time_Unet_FITS,
            'ModernTCN_Unet': ModernTCN_Unet,
            'Time_Unet_ModernBlock': Time_Unet_ModernBlock,
            'TimesNet': TimesNet,
            'Time_Unet_FPN': Time_Unet_FPN,
            'Time_Unet_PITS': Time_Unet_PITS
        }
        #初始化模型
        if self.args.model == 'ModernTCN_Unet':
            model = model_dict[self.args.model].Model(self.args.enc_in, self.args.seq_len, self.args.pred_len).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, setting):
        # train_loader(batch_sampler:30, batch_size:256, Dataset_ETT_hour:7873)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader) # 30
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        ##检查是否要使用混合精度训练,混合精度训练可以在一定程度上加快模型训练的速度，特别是在使用支持半精度加速的 GPU 上。可以在不影响模型精度的情况下减少训练时间和内存消耗
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        #通过这些参数，调度器将根据 One Cycle 策略自动调整学习率，并在训练过程中实现学习率的动态变化。
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        # Define Losses
        loss_D = nn.BCELoss()  # Binary Cross Entropy Loss for Generator
        loss_CD = nn.BCELoss()  # Binary Cross Entropy Loss for Central Discriminator

        # Initialize Generator, Discriminator, and Central Discriminator
        noise_len = 432  # Length of noise vector
        n_samples = 336  # Number of output samples
        alpha = 0.1  # Leaky ReLU slope
        n_channels = self.args.enc_in  # Number of channels
        gamma_value = 5.0
        generator = Generator(noise_len, n_samples, alpha)
        discriminator = Discriminator(n_samples, alpha)
        loss_function = nn.BCELoss()

        discriminators = {}
        for i in range(n_channels):
            discriminators[i] = Discriminator(n_samples=n_samples, alpha=alpha).apply(self.initialize_weights)
        for i in range(n_channels):
            discriminators[i] = discriminators[i].to("cuda:0")
            discriminators[i].to("cuda:0")
            ##

        generators = {}
        for i in range(n_channels):
            generators[i] = Generator(noise_len=noise_len, n_samples=n_samples, alpha=alpha).apply(self.initialize_weights)
        for i in range(n_channels):
            generators[i].to("cuda:0")

        gamma = [gamma_value] * self.args.train_epochs

        # Define Optimizers
        optimizers_D = {}
        for i in range(n_channels):
            optimizers_D[i] = torch.optim.Adam(discriminators[i].parameters(), lr=0.001, betas=[0.5, 0.9])
        optimizers_G = {}
        for i in range(n_channels):
            optimizers_G[i] = torch.optim.Adam(generators[i].parameters(), lr=0.001, betas=[0.5, 0.9])

        #central discriminator
        central_discriminator = Discriminator(n_samples=n_channels * n_samples, alpha=alpha)
        central_discriminator = central_discriminator.apply(self.initialize_weights)
        central_discriminator.to("cuda:0")
        optimizer_central_discriminator = torch.optim.Adam(central_discriminator.parameters(),
                                                           lr=0.0001, betas=[0.5, 0.9])


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            #将模型设置为训练模式，启用BatchNormalization和Dropout等训练相关的层
            self.model.train()
            self.eq.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                batch_x = batch_x.float().to(self.device) # bach_x:(256,432,7)
                batch_y = batch_y.float().to(self.device) # batch_y:(256,384,7)
                batch_x_mark = batch_x_mark.float().to(self.device) # batch_x_mark:(256,432,4)
                batch_y_mark = batch_y_mark.float().to(self.device) # batch_y_mark:(256,384,4)

                model_optim.zero_grad()
                self.equalizer_optimizer.zero_grad()
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() #(256,336,7) 全零
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) #(256,384,7)

                signal_group = {}
                for i in range(n_channels):
                    signal_group[i] = batch_x[:, :, i]
                # Generate noise
                #noise = torch.randn(batch_x.shape[0], n_channels, noise_len)
                shared_noise = torch.randn((batch_x.shape[0], noise_len)).float().to("cuda:0")

                # Generate fake data
               # generated_data = generator(noise).to("cuda:0")
                generated_samples = {}
                for i in range(n_channels):
                    generated_samples[i] = generators[i](shared_noise).float()

                generated_samples_labels = torch.zeros((batch_x.shape[0], n_samples)).to("cuda:0").float()
                real_samples_labels = torch.ones((batch_x.shape[0], n_samples)).to("cuda:0").float()
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Data for training the discriminators
                all_samples_group = {}
                for i in range(n_channels):
                    all_samples_group[i] = torch.cat(
                        (signal_group[i], generated_samples[i])
                    )

                # Training the discriminators
                outputs_D = {}
                loss_D = {}
                for i in range(n_channels):
                    optimizers_D[i].zero_grad()
                    outputs_D[i] = discriminators[i](all_samples_group[i].float())
                    loss_D[i] = loss_function(outputs_D[i], all_samples_labels)
                    loss_D[i].backward(retain_graph=True)
                    optimizers_D[i].step()

                #CD
                temp_generated = generated_samples[0]
                for i in range(1, n_channels):
                    temp_generated = torch.hstack((temp_generated, generated_samples[i]))
                group_generated = temp_generated

                temp_real = signal_group[0]
                for i in range(1, n_channels):
                    temp_real = torch.hstack((temp_real, signal_group[i]))
                group_real = temp_real

                all_samples_central = torch.cat((group_generated, group_real))
                all_samples_labels_central = torch.cat(
                    (torch.zeros((batch_x.shape[0], n_samples * n_channels)).to("cuda:0").float(), torch.ones((batch_x.shape[0], n_samples * n_channels)).to("cuda:0").float())
                )

                # Training the central discriminator
                optimizer_central_discriminator.zero_grad()
                output_central_discriminator = central_discriminator(all_samples_central.float())
                loss_central_discriminator = loss_function(
                    output_central_discriminator, all_samples_labels_central)
                loss_central_discriminator = loss_central_discriminator.clone().detach()
                loss_central_discriminator.requires_grad = True
                loss_central_discriminator.backward(retain_graph=True)
                optimizer_central_discriminator.step()

                # Training the generators
                outputs_G = {}
                loss_G_local = {}
                loss_G = {}
                for i in range(n_channels):
                    optimizers_G[i].zero_grad()
                    outputs_G[i] = discriminators[i](generated_samples[i])
                    loss_G_local[i] = loss_function(outputs_G[i], real_samples_labels)
                    all_samples_central_new = {}
                    output_central_discriminator_new = {}
                    loss_central_discriminator_new = {}

                    generated_samples_new = {}
                    for j in range(n_channels):
                        generated_samples_new[j] = generators[j](shared_noise)

                        if i == j:
                            generated_samples_new[j] = generated_samples_new[j].float()
                        else:
                            generated_samples_new[j] = generated_samples_new[j].detach().float()
                    temp_generated = generated_samples_new[0]
                    for j in range(1, n_channels):
                        temp_generated = torch.hstack((temp_generated, generated_samples_new[j]))
                    all_generated_samples = temp_generated
                    all_samples_central_new[i] = torch.cat((all_generated_samples, group_real))
                    output_central_discriminator_new[i] = central_discriminator(all_samples_central_new[i].float())
                    loss_central_discriminator_new[i] = loss_function(
                        output_central_discriminator_new[i], all_samples_labels_central)

                    loss_G[i] = loss_G_local[i] - gamma[epoch] * loss_central_discriminator_new[i]
                    loss_G[i].backward(retain_graph=True)
                    optimizers_G[i].step()

                # encoder - decoder
                #检查是否要使用混合精度训练,在不影响模型精度的情况下减少训练时间和内存消耗
                if self.args.use_amp:
                    with torch.cuda.amp.autocast(): #开启混合精度自动转换上下文
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention: #如果模型需要输出注意力权重
                                #这里使用了索引 [0] 是因为如果模型输出了注意力权重，那么在元组中的第一个位置。
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                         #   outputs = self.model(batch_x) # bach_x:(256,432,7)   outputs: (256,336,7),一堆小数点
                          #  outputs_aug = self.model(noise_injection(amplitude_perturbation(batch_x))) #数据增强
                            #引入SimPSI数据增强

                            data_psi, _, _, _ = self.augboost(batch_x, self.eq, batch_y, self.model)
                            outputs = self.model(data_psi)

                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            # batch_x(32,432,7) batch_x_mark(32,432,4) dec_inp(32,384,7) batch_y_mark(32,384,4) batch_y(32,384,7)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y) #(32,336,7)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] #(256,336,7)  (32,336,7)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) #(256,336,7)  (32,336,7)
                    loss = criterion(outputs, batch_y) * 0.5 + loss_central_discriminator * 0.5
                    # if outpus_psi.numel() != Null:
                    #     outputs_aug = outputs_aug[:, -self.args.pred_len:, f_dim:]
                    #     loss_aug = criterion(outputs_aug, batch_y)
                    #     loss = loss * 0.5 + loss_aug * 0.5
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    #优化器会根据当前设置的学习率和梯度计算出的参数更新值来更新模型的参数
                    model_optim.step()
                    self.equalizer_optimizer.step()
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #调用 early_stopping 对象的 __call__ 方法来判断是否需要提前停止训练，并根据需要保存模型
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Using TST---Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        test_flag = True
        tets_num = 0
        test_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    torch.cuda.synchronize()
                    start = time.time()
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    torch.cuda.synchronize()
                    end = time.time()
                    test_time = test_time + end - start
                    tets_num = tets_num + 1

                # 测试阶段统计模型的 FLOPs（浮点运算数）和参数数量，以及当前 GPU 占用的内存量，
                if test_flag:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Unet' in self.args.model:
                        # 使用 thopprofile 函数计算模型的 FLOPs 和参数数量
                        flops, params = thopprofile(self.model, inputs=(batch_x,))
                        print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
                        print("params=", str(params / 1e6) + '{}'.format("M"))
                        print(f'memary = {torch.cuda.memory_allocated() / 1024 / 1024}')
                        test_flag = False  # test_flag 设置为 False，确保这部分代码只在测试阶段执行一次。
                    else:
                        flops, params = thopprofile(self.model, inputs=(batch_x, batch_x_mark, dec_inp, batch_y_mark,))
                        print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
                        print("params=", str(params / 1e6) + '{}'.format("M"))
                        print(f'memary = {torch.cuda.memory_allocated() / 1024 / 1024}')
                        test_flag = False
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        print(f"avg_time = {test_time / tets_num}")
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        # 获取当前时间
        current_time = datetime.datetime.now()
        # 格式化时间字符串
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        f = open("result.txt", 'a')
        f.write('Time: {}\n'.format(time_str))  # 写入当前时间
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # with open(result_file_path, 'a') as f:
        #     f.write('Time: {}\n'.format(time_str)) #写入当前时间
        #     f.write(setting + "  \n")
        #     f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        #     f.write('\n')
        #     f.write('\n')

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
