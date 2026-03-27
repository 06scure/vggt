"""
光度立体任务评估脚本

用于评估训练好的VGGT模型在DiLiGenT_518测试集上的性能。
计算法向量预测的MAE（平均角度误差）。

使用示例:
    python eval.py --ckpt_path ckpt/ps_train/best_model.pth
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import argparse
import torch.nn.functional as F

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from training.data.datasets.diligent import DiLiGenTDataset
from training.ps_loss import PSLoss


def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def compute_normal_mae(pred_normal: torch.Tensor, gt_normal: torch.Tensor, mask: torch.Tensor) -> float:
    """
    计算法向量预测的平均角度误差（MAE）

    Args:
        pred_normal: 预测的法向量 [B, 3, H, W]
        gt_normal: 真实的法向量 [B, 3, H, W]
        mask: 掩码 [B, H, W]，值为True表示有效区域

    Returns:
        平均角度误差（度）
    """
    # 归一化预测法向量，确保是单位向量
    pred_normal_normalized = F.normalize(pred_normal, p=2, dim=1)
    gt_normal_normalized = F.normalize(gt_normal, p=2, dim=1)

    # 计算余弦相似度
    dot_product = torch.sum(pred_normal_normalized * gt_normal_normalized, dim=1)
    dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

    # 计算角度（弧度）并转换为度
    angles_rad = torch.acos(dot_product)
    angles_deg = angles_rad * 180.0 / np.pi

    # 应用掩码
    valid_angles = angles_deg[mask.to(torch.bool)]

    if valid_angles.numel() == 0:
        return 0.0

    return valid_angles.mean().item()


def evaluate_model(model, dataloader, criterion, device, logger):
    """
    评估模型性能

    Args:
        model: 要评估的模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        logger: 日志记录器

    Returns:
        平均损失和平均角度误差
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_normal_errors = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluation', leave=False)
        for batch in pbar:
            # 数据移到设备
            images = batch['images'].to(device)
            gt_normal = batch['normal'].to(device)
            mask = batch['mask'].to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            pred_dict = {'normal': predictions['normal']}
            batch_dict = {'normal': gt_normal, 'mask': mask}
            loss_dict = criterion(pred_dict, batch_dict)

            # 计算法向量误差
            mae = compute_normal_mae(predictions['normal'], gt_normal, mask)
            all_normal_errors.append(mae)

            # 记录损失
            total_loss += loss_dict['objective'].item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss_dict["objective"].item():.4f}', 'MAE': f'{mae:.2f}°'})

            # 清理显存
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = np.mean(all_normal_errors) if all_normal_errors else 0.0

    return avg_loss, avg_mae


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description="评估光度立体任务的VGGT模型")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="训练好的模型检查点路径"
    )
    parser.add_argument(
        "--img_per_seq",
        type=int,
        default=10,
        help="每个序列使用的图像数量（默认: 10）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批量大小（默认: 1，避免OOM）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器工作进程数（默认: 4）"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/ps_eval",
        help="日志目录（默认: logs/ps_eval）"
    )

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info("=" * 50)
    logger.info("Starting photometric stereo evaluation")
    logger.info(f"Configuration: {vars(args)}")

    # 创建数据集
    logger.info("Loading DiLiGenT_518 test dataset...")
    dataset = DiLiGenTDataset(
        data_dir='/home/user/dataset/DiLiGenT_518',
        img_size=518,
        img_per_seq=args.img_per_seq,
        split='test'
    )

    logger.info(f"Test dataset size: {len(dataset)}")

    # 创建数据加载器
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型
    logger.info("Creating model...")
    model = VGGT(
        img_size=518,
        enable_camera=False,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_normal=True
    )

    # 加载模型权重
    logger.info(f"Loading weights from {args.ckpt_path}")
    if args.ckpt_path.endswith('.pth') or args.ckpt_path.endswith('.pt'):
        # 加载完整检查点
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Checkpoint loaded. Epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        # 加载直接保存的模型
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    # 确保 aggregator 处于冻结状态（如果没有冻结）
    model.freeze_aggregator()

    model = model.to(device)

    # 创建损失函数
    criterion = PSLoss(normal={'weight': 1.0, 'loss_type': 'mse'})

    # 评估模型
    logger.info("=" * 50)
    logger.info("Evaluating model...")
    avg_loss, avg_mae = evaluate_model(model, test_loader, criterion, device, logger)

    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info(f"{'Average Loss':<20}: {avg_loss:.4f}")
    logger.info(f"{'MAE (degrees)':<20}: {avg_mae:.2f}")

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"MAE (degrees): {avg_mae:.2f}")


if __name__ == '__main__':
    main()