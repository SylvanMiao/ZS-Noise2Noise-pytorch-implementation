import argparse
import logging
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


def train_one_image(model, noisy_img, config, log_interval=200):
    """对单张噪声图从头训练，返回 loss 历史记录"""
    lr = config['lr']
    step_size = config['step_size']
    gamma = config['gamma']
    max_epoch = config['max_epoch']

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_history = []
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = loss_func(model, noisy_img)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logging.info(f'  epoch {epoch + 1:>5d}/{max_epoch}  loss={loss_val:.6f}  lr={scheduler.get_last_lr()[0]:.2e}')

    return loss_history


def save_tensor_as_image(tensor, path):
    """将 [0,1] 范围的 C×H×W 张量保存为 16-bit PNG 图片"""
    img = tensor.squeeze(0).clamp(0, 1).cpu()
    img = (img * 65535).round().numpy().astype(np.uint16)
    # img shape: (C, H, W)
    if img.shape[0] == 1:
        # 16-bit 灰度图: (1, H, W) -> (H, W)
        Image.fromarray(img[0], mode='I;16').save(path)
    else:
        # 16-bit RGB: (C, H, W) -> (H, W, C)
        Image.fromarray(img.transpose(1, 2, 0)).save(path)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ZS-N2N: Zero-Shot Noise2Noise')
    parser.add_argument('--config', type=str, default='parameters.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--data_path', type=str, default=None,
                        help='噪声图片文件夹路径（覆盖配置文件中的设置）')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='去噪结果输出目录')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='每隔多少 epoch 记录一次 loss')
    args = parser.parse_args()

    # 读取 YAML 配置文件中的超参数
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 命令行 --data_path 优先于配置文件中的 data_path
    data_path = args.data_path or config.get('data_path', '')
    if not data_path:
        raise ValueError('data_path must be set in parameters.yaml or via --data_path')

    # 命令行 --log_interval 优先于配置文件中的 log_interval，都未设置则默认 200
    log_interval = args.log_interval or config.get('log_interval', 200)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志配置：同时输出到控制台和文件
    log_path = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(),
        ],
    )

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    logging.info(f'Data path: {data_path}')
    logging.info(f'Config: max_epoch={config["max_epoch"]}, lr={config["lr"]}, '
                 f'step_size={config["step_size"]}, gamma={config["gamma"]}')

    # 加载数据集
    dataset = NoisyImageDataset(data_path)
    if len(dataset) == 0:
        raise RuntimeError(f'No images found in {data_path}')
    logging.info(f'Found {len(dataset)} image(s)')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 逐张处理
    for noisy_img, img_path in tqdm(dataloader, desc='Denoising'):
        noisy_img = noisy_img.to(device)
        in_channel = noisy_img.shape[1]
        name = Path(img_path[0]).stem

        logging.info(f'--- Processing: {name}  (shape={tuple(noisy_img.shape)}) ---')

        # 保存原始噪声图以便对比
        save_tensor_as_image(noisy_img, output_dir / f'{name}_noisy.png')

        # 为每张图new一个全新模型
        model = network(in_channel).to(device)
        loss_history = train_one_image(model, noisy_img, config, log_interval=log_interval)

        # 推理：干净图 = 噪声图 - 预测噪声
        with torch.no_grad():
            denoised = noisy_img - model(noisy_img)

        # log
        logging.info(f'  final_loss={loss_history[-1]:.6f}  '
                     f'noisy_min={noisy_img.min().item():.4f}  noisy_max={noisy_img.max().item():.4f}  '
                     f'denoised_min={denoised.min().item():.4f}  denoised_max={denoised.max().item():.4f}')

        save_tensor_as_image(denoised, output_dir / f'{name}_denoised.png')

    logging.info('Done.')


if __name__ == '__main__':
    main()
