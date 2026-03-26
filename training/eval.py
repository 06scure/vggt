import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
"""
python training/eval.py \
    --checkpoint_path ckpt/best_model.pt \
    --image_num 10
"""

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from training.data.datasets.ps_diligent import DiLiGenTDataset
from training.ps_loss import PSLoss


def evaluate_normal_accuracy(pred_normals, gt_normals, mask=None):
    """
    评估法向量预测的准确率（角度误差）

    Args:
        pred_normals: 预测的法向量 (C, H, W) 或 (B, C, H, W)，已归一化到单位长度
        gt_normals: 真实法向量 (C, H, W) 或 (B, C, H, W)，已归一化到单位长度
        mask: 有效区域掩码 (H, W) 或 (B, H, W)

    Returns:
        mean_angle: 平均角度误差（度）
        median_angle: 中位数角度误差（度）
        angular_error: 角度误差图
    """
    # 统一添加 batch 维度
    if pred_normals.dim() == 3:
        pred_normals = pred_normals.unsqueeze(0)
    if gt_normals.dim() == 3:
        gt_normals = gt_normals.unsqueeze(0)
    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # 计算角度误差（弧度）
    dot_product = torch.sum(pred_normals * gt_normals, dim=1).clamp(-1.0, 1.0)
    angular_error = torch.acos(dot_product)  # 弧度，形状 [B, H, W]

    if mask is not None:
        # 确保 mask 是布尔类型且形状匹配
        mask = mask.to(torch.bool)
        # 使用 mask 索引
        angular_error = angular_error[mask]
    else:
        angular_error = angular_error.flatten()

    # 转换为度
    angular_error_deg = angular_error * 180.0 / torch.pi

    # 计算统计信息
    mean_angle = torch.mean(angular_error_deg)
    median_angle = torch.median(angular_error_deg)

    return mean_angle.item(), median_angle.item(), angular_error_deg


def save_normal_images(pred_normals, output_dir, seq_name, mask=None):
    """
    保存法向量可视化图像

    Args:
        pred_normals: 预测的法向量 (C, H, W)，范围[-1, 1]
        output_dir: 输出目录
        seq_name: 序列名称
        mask: 有效区域掩码 (H, W)，背景区域将显示为黑色
    """
    os.makedirs(output_dir, exist_ok=True)

    # 转换到[0, 255]范围
    normal_img = ((pred_normals + 1.0) / 2.0 * 255.0).permute(1, 2, 0).cpu().numpy()

    # 确保数值在有效范围内
    normal_img = np.clip(normal_img, 0, 255).astype(np.uint8)

    # 法向量通常使用RGB顺序保存（OpenCV imwrite默认使用BGR，所以需要转换）
    # 注意：法向量可视化通常RGB分别对应X,Y,Z分量，不需要特别调整
    # normal_img = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)

    # 如果有mask，将背景区域设为黑色
    if mask is not None:
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        normal_img[~mask_np] = 0

    save_path = os.path.join(output_dir, f'{seq_name}_pred_normal.png')
    cv2.imwrite(save_path, normal_img)
    logging.info(f'保存法向量图像: {save_path}')


def evaluate(model, dataset, device, output_dir=None):
    """
    评估模型在DiLiGenT数据集上的性能

    Args:
        model: 训练好的模型
        dataset: DiLiGenT数据集
        device: 设备
        output_dir: 输出目录

    Returns:
        mean_angle: 平均角度误差
        median_angle: 中位数角度误差
    """
    model.eval()
    total_mean_angle = 0.0
    total_median_angle = 0.0
    count = 0

    criterion = PSLoss()

    with torch.no_grad():
        for seq_name in dataset.sequence_list:
            logging.info(f'评估序列: {seq_name}')

            # 获取序列数据
            seq_data = dataset.get_data(seq_name=seq_name)

            # 转换为tensor
            images = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in seq_data['images']]
            images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # B, S, C, H, W

            gt_normal = seq_data['gt_normal']
            if gt_normal is not None:
                # PSDataset.get_data 已经解码好了，直接使用
                gt_normal = torch.from_numpy(gt_normal).permute(2, 0, 1).float()
                # 确保法向量是单位长度
                gt_normal = F.normalize(gt_normal, p=2, dim=0, eps=1e-6)
                gt_normal = gt_normal.to(device)

            mask = seq_data['mask']
            if mask is not None:
                mask = torch.from_numpy(mask).bool().to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            batch_data = {'gt_normal': gt_normal.unsqueeze(0), 'mask': mask.unsqueeze(0)}
            loss_dict = criterion(predictions, batch_data)

            # 评估角度误差
            if 'normals' in predictions:
                mean_angle, median_angle, _ = evaluate_normal_accuracy(
                    predictions['normals'],
                    gt_normal,
                    mask
                )

                total_mean_angle += mean_angle
                total_median_angle += median_angle
                count += 1

                logging.info(f'  角度误差: 平均 {mean_angle:.2f} 度, 中位数 {median_angle:.2f} 度')

                # 保存法向量图像
                if output_dir is not None:
                    save_normal_images(
                        predictions['normals'][0],
                        os.path.join(output_dir, 'normal_images'),
                        seq_name,
                        mask
                    )

    if count > 0:
        avg_mean_angle = total_mean_angle / count
        avg_median_angle = total_median_angle / count
        logging.info(f'平均角度误差: {avg_mean_angle:.2f} 度')
        logging.info(f'中位数角度误差: {avg_median_angle:.2f} 度')
        return avg_mean_angle, avg_median_angle

    return None, None


def main():
    parser = argparse.ArgumentParser(description='评估光度立体法向量预测模型')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--img_size', type=int, default=518, help='图像大小')
    parser.add_argument('--output_dir', type=str, default='./logs/eval', help='输出目录')
    parser.add_argument('--image_num', type=int, default='10', help='推理图像数量')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 设备
    device = torch.device(args.device)

    # 创建数据集
    class CommonConfig:
        img_size: int = args.img_size
        training: bool = False
        patch_size: int = 14
        class Augs:
            scales: list = [0.8, 1.2]
        augs: Augs = Augs()
        rescale: bool = False
        rescale_aug: bool = False
        landscape_check: bool = False
        debug: bool = False
        get_nearby: bool = False
        load_depth: bool = False
        inside_random: bool = False
        allow_duplicate_img: bool = False

    common_conf = CommonConfig()
    dataset = DiLiGenTDataset(common_conf=common_conf, split='test', img_per_seq=args.image_num)

    # 创建模型
    model = VGGT(
        img_size=args.img_size,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        enable_normal=True
    )

    # 加载模型权重
    logging.info(f'加载模型权重: {args.checkpoint_path}')
    if torch.cuda.is_available():
        state_dict = torch.load(args.checkpoint_path)['model_state_dict']
    else:
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')['model_state_dict']

    model.load_state_dict(state_dict)
    model.to(device)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 评估
    logging.info('开始评估')
    avg_mean_angle, avg_median_angle = evaluate(model, dataset, device, args.output_dir)

if __name__ == '__main__':
    main()