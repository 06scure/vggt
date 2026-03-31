"""
光度立体任务训练脚本 (版本2)

支持多帧法向量预测 + 不确定性损失

使用示例:
    python train_v2.py
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import swanlab


# 设置环境变量以优化PyTorch内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from training.ps_loss_v2 import PSLossV2
from training.data.datasets.wild import PSWildDataset
from training.data.datasets.diligent import DiLiGenTDataset


def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, val_mae, ckpt_path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_mae': val_mae,
        'config': config,
    }, ckpt_path)


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_normal_errors = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for batch in pbar:
            # 数据移到设备
            images = batch['images'].to(device)
            gt_normal = batch['normal'].to(device)
            mask = batch['mask'].to(device)

            # 前向传播
            predictions = model(images)

            # 在验证时，我们需要手动融合多帧预测
            from vggt.heads.normal_head_v2 import fuse_normals_with_confidence
            normal_all = predictions['normal_all']
            normal_conf = predictions['normal_conf']
            pred_normal = fuse_normals_with_confidence(normal_all, normal_conf, mask=mask)

            # 计算损失 - 为了验证，我们还是构造一个包含'normal'的字典
            # 注意：这里的损失计算只用融合后的结果，主要是为了监控
            pred_dict = {
                'normal_all': predictions['normal_all'],
                'normal_conf': predictions['normal_conf'],
                'normal': pred_normal  # 添加融合后的结果
            }
            batch_dict = {'normal': gt_normal, 'mask': mask}

            # 计算损失（只用per-frame loss，不用fused loss）
            # 临时修改criterion的use_fused_loss
            orig_use_fused = criterion.use_fused_loss
            criterion.use_fused_loss = False
            loss_dict = criterion(pred_dict, batch_dict)
            criterion.use_fused_loss = orig_use_fused

            # 计算角度误差（MAE）
            valid_mask = mask.to(torch.bool)

            # 计算余弦相似度
            dot_product = torch.sum(pred_normal * gt_normal, dim=1)
            dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
            angle_error = torch.acos(dot_product) * 180.0 / np.pi  # 转换为角度

            # 只计算有效区域的误差
            if valid_mask.sum() > 0:
                mean_angle_error = angle_error[valid_mask].mean().item()
                all_normal_errors.append(mean_angle_error)

            total_loss += loss_dict['objective'].item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss_dict["objective"].item():.4f}', 'MAE': f'{mean_angle_error:.2f}°'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = np.mean(all_normal_errors) if all_normal_errors else 0.0

    return avg_loss, avg_mae


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
    logger
):
    """训练主函数 - 按step控制保存和验证"""
    model.train()

    best_mae = float('inf')
    global_step = 0
    accum_steps = config['accum_steps']

    # 清零梯度
    optimizer.zero_grad()

    for epoch in range(1, config['max_epochs'] + 1):
        logger.info(f"Starting Epoch {epoch}/{config['max_epochs']}")

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, batch in enumerate(pbar):
            current_step = global_step + batch_idx + 1

            # 数据移到设备
            images = batch['images'].to(device)  # [B, N, 3, H, W]
            gt_normal = batch['normal'].to(device)  # [B, 3, H, W]
            mask = batch['mask'].to(device)  # [B, H, W]

            # 前向传播 - 使用BF16混合精度
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config['use_bf16']):
                predictions = model(images)

                # 构建损失计算所需的字典
                pred_dict = {
                    'normal_all': predictions['normal_all'],
                    'normal_conf': predictions['normal_conf']
                }
                batch_dict = {'normal': gt_normal, 'mask': mask}

                # 计算损失
                loss_dict = criterion(pred_dict, batch_dict)
                loss = loss_dict['objective'] / accum_steps

            # 反向传播
            loss.backward()

            # 梯度累积
            if (batch_idx + 1) % accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_max_norm'])
                optimizer.step()
                optimizer.zero_grad()

            # 更新进度条
            postfix = {'loss': f'{loss_dict["objective"].item():.4f}', 'step': current_step}
            if 'loss_normal_data' in loss_dict:
                postfix['data'] = f'{loss_dict["loss_normal_data"].item():.4f}'
            if 'loss_normal_uncertainty' in loss_dict:
                postfix['uncert'] = f'{loss_dict["loss_normal_uncertainty"].item():.4f}'
            pbar.set_postfix(postfix)

            # 记录到swanlab
            current_lr = optimizer.param_groups[0]['lr']
            log_dict = {
                "train/loss": loss_dict["objective"].item(),
                "lr": current_lr
            }
            if 'loss_normal_data' in loss_dict:
                log_dict["train/loss_data"] = loss_dict["loss_normal_data"].item()
            if 'loss_normal_uncertainty' in loss_dict:
                log_dict["train/loss_uncertainty"] = loss_dict["loss_normal_uncertainty"].item()

            swanlab.log(log_dict, step=current_step)

            # ========== 按step验证和保存 ==========
            if current_step % config['save_step_freq'] == 0:
                logger.info(f"Step {current_step}: Running validation...")

                # 验证
                val_loss, val_mae = validate(model, val_loader, criterion, device)
                logger.info(f"Step {current_step} - Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}°")

                # 记录到swanlab
                swanlab.log({
                    "val/loss": val_loss,
                    "val/mae": val_mae
                }, step=current_step)

                # 保存定期检查点
                ckpt_path = os.path.join(config['ckpt_dir'], f'checkpoint_step_{current_step}.pt')
                save_checkpoint(
                    model, optimizer, scheduler, epoch, current_step,
                    config, val_mae, ckpt_path
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")

                # 保存最佳模型
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_ckpt_path = os.path.join(config['ckpt_dir'], 'best_model_v2.pt')
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, current_step,
                        config, val_mae, best_ckpt_path
                    )
                    logger.info(f"Saved best model to {best_ckpt_path} (MAE: {val_mae:.2f}°)")

                # 回到训练模式
                model.train()

        # 更新全局step
        global_step += len(train_loader)

        # 更新学习率（每个epoch结束时）
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch} completed - LR: {current_lr:.8f}")

        # 每个epoch结束也保存一下（防止step不是正好对齐）
        epoch_ckpt_path = os.path.join(config['ckpt_dir'], f'checkpoint_epoch_{epoch}.pt')
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            config, float('inf'), epoch_ckpt_path
        )
        logger.info(f"Saved epoch checkpoint to {epoch_ckpt_path}")

        # 清理显存
        torch.cuda.empty_cache()

    logger.info("=" * 50)
    logger.info(f"Training completed! Best MAE: {best_mae:.2f}°")

    return best_mae


def main():
    """主训练函数"""
    # 配置参数
    config = {
        'batch_size': 1,
        'num_workers': 8,
        'img_per_seq': 10,
        'learning_rate': 3e-6,
        'weight_decay': 0.05,
        'max_epochs': 6,
        'save_step_freq': 2000,  # 每2000 step保存和验证一次
        'accum_steps': 2,
        'log_dir': 'logs/ps_train_v21',
        'ckpt_dir': 'ckpt/ps_train_v21',
        'pretrained_ckpt': '/home/user/dataset/ckpt/model.pt',
        'loss_type': 'mse',
        'uncertainty_weight': 0.03,
        'use_bf16': True,
        'grad_clip_max_norm': 1.0,
    }

    # 设置日志
    logger = setup_logging(config['log_dir'])

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 初始化swanlab
    swanlab.init(
            project="ps-vggt",
            name=f"ps_train_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Photometric Stereo training with VGGT (multi-frame + uncertainty)",
            config=config
        )

    logger.info("=" * 50)
    logger.info("Starting photometric stereo training (v2 - multi-frame + uncertainty)")
    logger.info(f"Configuration: {config}")

    # 创建检查点目录
    os.makedirs(config['ckpt_dir'], exist_ok=True)

    # 创建数据集
    logger.info("Loading datasets...")
    train_dataset = PSWildDataset(
        data_dir='/home/user/dataset/PSWild',
        img_size=504,
        img_per_seq=config['img_per_seq'],
        split='train'
    )

    val_dataset = DiLiGenTDataset(
        data_dir='/home/user/dataset/DiLiGenT_518',
        img_size=518,
        img_per_seq=10,
        split='test'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

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

    # 加载预训练权重
    if os.path.exists(config['pretrained_ckpt']):
        logger.info(f"Loading pretrained weights from {config['pretrained_ckpt']}")
        state_dict = torch.load(config['pretrained_ckpt'], map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Pretrained checkpoint not found at {config['pretrained_ckpt']}")

    # 冻结aggregator，只训练normal_head
    logger.info("Freezing aggregator weights...")
    model.freeze_aggregator()

    # 打印可训练参数
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {num_trainable:,} / {num_total:,}")

    model = model.to(device)

    # 创建损失函数
    criterion = PSLossV2(
        normal={'weight': 1.0, 'loss_type': config['loss_type']},
        uncertainty_weight=config['uncertainty_weight'],
        use_fused_loss=False,
        use_per_frame_loss=True
    )

    # 创建优化器
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['max_epochs'],
        eta_min=1e-7
    )

    # 开始训练
    best_mae = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger
    )

    # 关闭swanlab
    swanlab.finish()


if __name__ == '__main__':
    main()