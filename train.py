import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import NoisyImageDataset
from model import network
from transforms.pair_downsample import pair_downsampler


def mseloss(gt, pred):
    """计算两张图像之间的 MSE 损失"""
    return nn.functional.mse_loss(gt, pred)


def loss_func(model, noisy_img):
    """
    ZS-N2N 对称一致性损失
    对噪声图做双下采样得到两个子图，利用模型预测噪声残差，
    通过残差一致性和重建损失约束训练
    """
    # 双下采样：将噪声图分为两个互补的子图
    noisy1, noisy2 = pair_downsampler(noisy_img)

    # 模型预测两个子图的噪声残差
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    # 重建损失：交叉约束（子图1的残差应能去噪子图2）
    loss_res = 0.5 * mseloss(noisy1, pred2) + 0.5 * mseloss(noisy2, pred1)

    # 一致性损失：全图的残差下采样后应与子图残差一致
    denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(denoised)

    loss_cons = 0.5 * mseloss(pred2, denoised2) + 0.5 * mseloss(pred1, denoised1)

    return loss_res + loss_cons


def train_one_image(model, noisy_img, config):
    """对单张噪声图从头训练"""
    lr = config['lr']
    step_size = config['step_size']
    gamma = config['gamma']
    max_epoch = config['max_epoch']

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for _ in range(max_epoch):
        optimizer.zero_grad()
        loss = loss_func(model, noisy_img)
        loss.backward()
        optimizer.step()
        scheduler.step()


def save_tensor_as_image(tensor, path):
    """将 [0,1] 范围的 C×H×W 张量保存为 PNG 图片，兼容灰度图和 RGB 图"""
    img = tensor.squeeze(0).clamp(0, 1).cpu()
    img = (img * 255).numpy().astype(np.uint8)
    if img.ndim == 2:
        # 灰度图 (H, W)
        Image.fromarray(img, mode='L').save(path)
    else:
        # RGB 图 (C, H, W) → (H, W, C)
        img = img.transpose(1, 2, 0)
        Image.fromarray(img).save(path)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ZS-N2N: Zero-Shot Noise2Noise')
    parser.add_argument('--config', type=str, default='parameters.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--data_path', type=str, default=None,
                        help='噪声图片文件夹路径（覆盖配置文件中的设置）')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='去噪结果输出目录')
    args = parser.parse_args()

    # 读取 YAML 配置文件中的超参数
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 命令行 --data_path 优先于配置文件中的 data_path
    data_path = args.data_path or config.get('data_path', '')
    if not data_path:
        raise ValueError('data_path must be set in parameters.yaml or via --data_path')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动选择 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Data path: {data_path}')
    print(f'Config: max_epoch={config["max_epoch"]}, lr={config["lr"]}, '
          f'step_size={config["step_size"]}, gamma={config["gamma"]}')

    # 加载数据集
    dataset = NoisyImageDataset(data_path)
    if len(dataset) == 0:
        raise RuntimeError(f'No images found in {data_path}')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 逐张处理
    for noisy_img, img_path in tqdm(dataloader, desc='Denoising'):
        noisy_img = noisy_img.to(device)
        in_channel = noisy_img.shape[1]

        # 为每张图new一个全新模型
        model = network(in_channel).to(device)
        train_one_image(model, noisy_img, config)

        # 推理：干净图 = 噪声图 - 预测噪声
        with torch.no_grad():
            denoised = noisy_img - model(noisy_img)

        name = Path(img_path[0]).stem
        save_path = output_dir / f'{name}_denoised.png'
        save_tensor_as_image(denoised, save_path)

    print('Done.')


if __name__ == '__main__':
    main()
