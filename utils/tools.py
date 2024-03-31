import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        #'type1'：使用指数衰减的学习率调整方式，每个 epoch 学习率减少一半
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        #type2'：在指定的 epoch 处使用预定义的学习率
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        #'type3'：前三个 epoch 保持初始学习率不变，之后每个 epoch 学习率乘以0.9。
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        #'constant'：学习率保持不变。
        lr_adjust = {epoch: args.learning_rate}
    #'3', '4', '5', '6'：在指定的 epoch 处将学习率乘以0.1。
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        #'TST'：使用 scheduler 中的最后一个学习率。
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    #手动更改学习率
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('epoch=',epoch,'args.lradj=',args.lradj,'Updating learning rate to',lr)

##这段代码是一个用于提前停止模型训练的类 EarlyStopping。它的作用是在验证集上的损失函数值不再减小时提前终止模型的训练，以避免过拟合
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience  # patience: 表示容忍的验证集损失函数值不再减小的次数阈值，默认为7。
        self.verbose = verbose  # verbose: 控制是否打印提示信息，默认为False。
        self.counter = 0  # counter: 记录连续验证集损失函数值未减小的次数。
        self.best_score = None  # best_score: 记录最佳的验证集损失函数值。
        self.early_stop = False  # early_stop: 标记是否提前停止训练。
        self.val_loss_min = np.Inf  # val_loss_min: 记录最小的验证集损失函数值。
        self.delta = delta  # delta: 设置提前停止的阈值，即验证集损失函数值的变化量必须大于该值才会被认为有所改善，默认为0。

    def __call__(self, val_loss, model, path):
        score = -val_loss
        # 比较当前的验证集损失函数值与历史最佳值，如果当前值更好，则保存当前模型并更新历史最佳值
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        # 如果当前值没有达到期望的改善（即变化量小于 delta），则增加计数器，当计数器达到设定的 patience 阈值时，标记为需要提前停止训练
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # 如果当前值有所改善，则重置计数器，并保存当前模型
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))