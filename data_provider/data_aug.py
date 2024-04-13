import torch
import torch.fft as fft


def amplitude_perturbation(data, perturbation_factor=0.1):
    """
    幅度扰动：对时序数据中的数值进行随机的增减
    """
    perturbed_data = data * (1 + perturbation_factor * torch.randn_like(data))
    return perturbed_data

def noise_injection(data, noise_level=0.1):
    """
    噪声注入：向时序数据中添加随机噪声
    """
    noise = noise_level * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data


def clip_and_scale(data, start_percentile=10, end_percentile=90, scale_factor=1.0):
    """
    剪切和缩放：对时序数据的一部分进行剪切或缩放操作
    """
    # 计算时间步的百分位对应的索引
    start_idx = int(
        torch.kthvalue(torch.arange(data.shape[1]).float(), int(data.shape[1] * start_percentile / 100)).values)
    end_idx = int(torch.kthvalue(torch.arange(data.shape[1]).float(), int(data.shape[1] * end_percentile / 100)).values)
    # 复制数据，以免修改原始数据
    clipped_scaled_data = data.clone()
    # 对指定部分进行剪切和缩放
    clipped_scaled_data[:, start_idx:end_idx, :] *= scale_factor

    return clipped_scaled_data


# 示例数据
batch_size = 256
time_steps = 96
channels = 7
time_series_data = torch.randn(batch_size, time_steps, channels)

perturbed_data = amplitude_perturbation(time_series_data, perturbation_factor=0.1)
noisy_data = noise_injection(time_series_data, noise_level=0.1)
clipped_scaled_data = clip_and_scale(time_series_data, start_percentile=10, end_percentile=90, scale_factor=1.5)


print(perturbed_data.shape)
print(noisy_data.shape)
print(clipped_scaled_data .shape)

