import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    import swanlab
    has_swanlab = True
except ImportError:
    has_swanlab = False

from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from training.data.datasets.ps_wild import PSWildDataset
from training.ps_loss import PSLoss

"""
python training/train.py \
    --batch_size 1 \
    --num_epochs 1 \
    --img_per_seq 8 \
    --checkpoint_path /home/user/dataset/ckpt/model.pt \
    --num_train_items 1000
"""
@dataclass
class CommonConfig:
    """通用配置类"""
    img_size: int = 518
    patch_size: int = 14
    augs: dict = None
    rescale: bool = True
    rescale_aug: bool = False
    landscape_check: bool = False
    training: bool = True
    debug: bool = False
    get_nearby: bool = False
    load_depth: bool = False
    inside_random: bool = False
    allow_duplicate_img: bool = False

    def __post_init__(self):
        if self.augs is None:
            self.augs = type('obj', (object,), {'scales': [1.0, 1.0]})()


class ToTensor(object):
    """将numpy数组转换为torch张量"""
    def __call__(self, sample):
        image, gt_normal, mask = sample['images'], sample['gt_normal'], sample['mask']

        # 转换图像: HWC -> CHW
        image = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in image]
        image = torch.stack(image, dim=0)  # S, C, H, W

        # 转换法向量: HWC -> CHW（PSDataset.get_data已经解码好了）
        if gt_normal is not None:
            gt_normal = torch.from_numpy(gt_normal).permute(2, 0, 1).float()
            # 确保法向量是单位长度
            if gt_normal.dim() == 3:
                gt_normal = F.normalize(gt_normal, p=2, dim=0, eps=1e-8)

        # 转换mask
        if mask is not None:
            mask = torch.from_numpy(mask).bool()

        return {
            'images': image,
            'gt_normal': gt_normal,
            'mask': mask,
            'seq_name': sample['seq_name'],
        }


def train_one_epoch(model, dataset, criterion, optimizer, device, epoch, swanlab_run, num_train_items=None):
    """
    训练一个epoch

    Args:
        model: 模型
        dataset: 数据集
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        num_train_items: 训练的样本数量（None表示全部）

    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    if num_train_items is not None:
        num_samples = min(num_train_items, len(dataset))
    else:
        num_samples = len(dataset)

    pbar = tqdm(range(num_samples), desc=f'Epoch {epoch}')
    for batch_idx in pbar:
        # 获取数据
        batch = dataset[batch_idx]

        # 将数据转换为tensor并移动到设备
        images = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in batch['images']]
        images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # B, S, C, H, W

        gt_normal = None
        if batch['gt_normal'] is not None:
            gt_normal = torch.from_numpy(batch['gt_normal']).permute(2, 0, 1).float()
            # PSDataset.get_data已经解码好了，直接确保是单位长度
            if gt_normal.dim() == 3:
                gt_normal = F.normalize(gt_normal, p=2, dim=0, eps=1e-6)
            gt_normal = gt_normal.unsqueeze(0).to(device)

        mask = None
        if batch['mask'] is not None:
            mask = torch.from_numpy(batch['mask']).bool().unsqueeze(0).to(device)

        # 前向传播
        predictions = model(images)

        # 计算损失
        batch_data = {'gt_normal': gt_normal, 'mask': mask}
        loss_dict = criterion(predictions, batch_data)
        loss = loss_dict['loss_normal'] if 'loss_normal' in loss_dict else 0.0

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计
        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

        # 实时记录到 SwanLab
        if swanlab_run is not None:
            # 计算当前平均损失
            current_avg_loss = total_loss / num_batches
            
            # 记录指标
            swanlab_run.log({
                "loss": loss.item(),
                "avg_loss": current_avg_loss,
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "lr": optimizer.param_groups[0]['lr']
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='训练光度立体法向量预测模型')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小 (注意：显卡为5070ti(16G)，要注意可能会OOM)')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--img_size', type=int, default=518, help='图像大小')
    parser.add_argument('--img_per_seq', type=int, default=8, help='每个序列的图像数量')
    parser.add_argument('--freeze_aggregator', action='store_true', default=True, help='是否冻结aggregator')
    parser.add_argument('--checkpoint_path', type=str, default='/home/user/dataset/ckpt/model.pt', help='预训练权重路径')
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--num_train_items', type=int, default=None, help='训练的样本数量（None表示全部）')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # SwanLab 实验初始化
    swanlab_run = None
    if has_swanlab:
        swanlab_run = swanlab.init(
            project="ps-vggt",
            experiment_name=f"train_{args.img_per_seq}imgs",
            config={
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_epochs": args.num_epochs,
                "img_size": args.img_size,
                "img_per_seq": args.img_per_seq,
                "freeze_aggregator": args.freeze_aggregator,
                "num_train_items": args.num_train_items,
            }
        )

    # 设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 创建数据集
    common_conf = CommonConfig(
        img_size=args.img_size,
        training=True
    )

    dataset = PSWildDataset(
        common_conf=common_conf,
        split='train',
        img_per_seq=args.img_per_seq
    )

    # 创建模型
    model = VGGT(
        img_size=args.img_size,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        enable_normal=True
    )

    # 加载预训练权重
    if os.path.exists(args.checkpoint_path):
        logger.info(f'加载预训练权重: {args.checkpoint_path}')
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        # 移除normal_head相关的键（如果有的话）
        state_dict = {k: v for k, v in state_dict.items() if 'normal_head' not in k}
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f'预训练权重文件不存在: {args.checkpoint_path}')

    # 冻结aggregator
    if args.freeze_aggregator:
        logger.info('冻结Aggregator权重')
        model.freeze_aggregator()

    # 移动模型到设备
    model = model.to(device)

    # 创建损失函数
    criterion = PSLoss()

    # 创建优化器（只训练需要梯度的参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # 训练循环
    logger.info('开始训练')
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        # 训练
        avg_loss = train_one_epoch(model, dataset, criterion, optimizer, device, epoch, swanlab_run, num_train_items=args.num_train_items)

        logger.info(f'Epoch {epoch}/{args.num_epochs}, 平均损失: {avg_loss:.6f}')

        # 保存检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_model_path)
            logger.info(f'保存最佳模型: {best_model_path}')

        # 定期保存
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'保存检查点: {checkpoint_path}')

        # 每个 epoch 结束后记录平均损失
        if swanlab_run is not None:
            swanlab_run.log({
                "epoch_avg_loss": avg_loss,
                "epoch": epoch + 1
            })

    logger.info('训练完成')
    swanlab_run.finish()


if __name__ == '__main__':
    main()