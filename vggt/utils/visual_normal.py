"""
法向量可视化工具

提供法向量预测结果的可视化功能：
- 左侧：真值法向量图
- 中间：预测法向量图
- 右侧：误差热力图

使用示例:
    from vggt.utils.visual_normal import visualize_normal_comparison

    # 可视化单个样本
    fig = visualize_normal_comparison(
        gt_normal=gt_normal,  # [3, H, W] or [H, W, 3]
        pred_normal=pred_normal,  # [3, H, W] or [H, W, 3]
        mask=mask,  # [H, W]
        save_path='comparison.png'
    )

    # 批量可视化
    from vggt.utils.visual_normal import visualize_batch_comparison
    visualize_batch_comparison(
        gt_normals=batch_gt,  # [B, 3, H, W]
        pred_normals=batch_pred,  # [B, 3, H, W]
        masks=batch_masks,  # [B, H, W]
        save_dir='vis_results'
    )
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Tuple
import matplotlib.colors as mcolors


def normal_to_rgb(normal_map: np.ndarray) -> np.ndarray:
    """
    将法向量图转换为RGB可视化图像

    Args:
        normal_map: 法向量图 [H, W, 3]，值域 [-1, 1]

    Returns:
        RGB图像 [H, W, 3]，值域 [0, 1]
    """
    # 归一化到单位长度（确保可视化正确）
    norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    # 避免除零，直接使用broadcasting
    normalized = normal_map / np.maximum(norm, 1e-8)

    # 从 [-1, 1] 转换到 [0, 1]
    return (normalized + 1) / 2


def compute_angle_error(
    gt_normal: np.ndarray,
    pred_normal: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算预测法向量与真值之间的角度误差（度）

    Args:
        gt_normal: 真值法向量 [H, W, 3]
        pred_normal: 预测法向量 [H, W, 3]
        mask: 掩码 [H, W]，True表示有效区域

    Returns:
        角度误差图 [H, W]，单位为度
    """
    # 归一化
    def normalize(n):
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / np.maximum(norm, 1e-8)

    gt_norm = normalize(gt_normal)
    pred_norm = normalize(pred_normal)

    # 计算余弦相似度
    dot_product = np.sum(gt_norm * pred_norm, axis=-1)
    dot_product = np.clip(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

    # 计算角度误差
    angles_rad = np.arccos(dot_product)
    angles_deg = angles_rad * 180.0 / np.pi

    # 应用掩码，无效区域设为NaN
    if mask is not None:
        angles_deg[~mask.astype(bool)] = np.nan

    return angles_deg


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """将tensor转换为numpy数组"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def visualize_normal_comparison(
    gt_normal: Union[torch.Tensor, np.ndarray],
    pred_normal: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
    error_max: float = 30.0,
    show: bool = True
) -> plt.Figure:
    """
    可视化法向量对比：左侧真值，中间预测，右侧误差热力图

    Args:
        gt_normal: 真值法向量，支持格式:
            - [3, H, W] (CHW)
            - [H, W, 3] (HWC)
        pred_normal: 预测法向量，格式同gt_normal
        mask: 掩码，[H, W]，True表示有效区域
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        figsize: 图像大小 (width, height)
        error_max: 误差热力图的最大值（度），超过此值的将被截断
        show: 是否调用plt.show()

    Returns:
        matplotlib Figure对象
    """
    # 转换为numpy
    gt = to_numpy(gt_normal)
    pred = to_numpy(pred_normal)
    if mask is not None:
        mask = to_numpy(mask)

    # 调整通道顺序：CHW -> HWC
    if gt.ndim == 3 and gt.shape[0] == 3:
        gt = np.transpose(gt, (1, 2, 0))
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))

    # 确保mask是2D
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.squeeze()

    # 计算角度误差
    angle_error = compute_angle_error(gt, pred, mask)

    # 转换为RGB用于可视化
    gt_rgb = normal_to_rgb(gt)
    pred_rgb = normal_to_rgb(pred)

    # 应用mask到RGB图像（背景设为黑色）
    if mask is not None:
        mask_bool = mask.astype(bool)
        gt_rgb[~mask_bool] = 0.0
        pred_rgb[~mask_bool] = 0.0

    # 创建图像
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 左侧：真值
    ax_gt = axes[0]
    ax_gt.imshow(gt_rgb)
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax_gt.axis('off')

    # 中间：预测
    ax_pred = axes[1]
    ax_pred.imshow(pred_rgb)
    ax_pred.set_title('Prediction', fontsize=14, fontweight='bold')
    ax_pred.axis('off')

    # 右侧：误差热力图
    ax_error = axes[2]
    # 使用jet colormap，但NaN（无效区域）设为黑色
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='black')

    # 绘制热力图
    im = ax_error.imshow(
        angle_error,
        cmap=cmap,
        vmin=0,
        vmax=error_max
    )
    ax_error.set_title('Angle Error (degrees)', fontsize=14, fontweight='bold')
    ax_error.axis('off')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax_error, fraction=0.046, pad=0.04)
    cbar.set_label('Error (°)', fontsize=12)

    # 计算并显示统计信息
    if mask is not None:
        valid_errors = angle_error[mask.astype(bool)]
    else:
        valid_errors = angle_error[~np.isnan(angle_error)]

    if len(valid_errors) > 0:
        mean_err = np.mean(valid_errors)
        median_err = np.median(valid_errors)
        stats_text = f'Mean: {mean_err:.2f}°\nMedian: {median_err:.2f}°'
        ax_error.text(
            0.02, 0.98, stats_text,
            transform=ax_error.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

    # 主标题
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    # 保存
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_batch_comparison(
    gt_normals: Union[torch.Tensor, np.ndarray],
    pred_normals: Union[torch.Tensor, np.ndarray],
    masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_dir: Union[str, Path] = 'vis_results',
    prefix: str = 'sample',
    max_samples: Optional[int] = None,
    **kwargs
) -> None:
    """
    批量可视化法向量对比结果

    Args:
        gt_normals: batch真值法向量 [B, 3, H, W]
        pred_normals: batch预测法向量 [B, 3, H, W]
        masks: batch掩码 [B, H, W]
        save_dir: 保存目录
        prefix: 文件名前缀
        max_samples: 最多可视化多少个样本，None表示全部
        **kwargs: 传递给visualize_normal_comparison的参数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = gt_normals.shape[0]
    if max_samples is not None:
        batch_size = min(batch_size, max_samples)

    for i in range(batch_size):
        gt = gt_normals[i]
        pred = pred_normals[i]
        mask = masks[i] if masks is not None else None

        save_path = save_dir / f'{prefix}_{i:03d}.png'

        visualize_normal_comparison(
            gt_normal=gt,
            pred_normal=pred,
            mask=mask,
            save_path=save_path,
            title=f'Sample {i}',
            show=False,
            **kwargs
        )

    plt.close('all')
    print(f'✓ 批量可视化完成，共保存 {batch_size} 张图片到 {save_dir}')


def visualize_confidence_comparison(
    gt_normal: Union[torch.Tensor, np.ndarray],
    pred_normal: Union[torch.Tensor, np.ndarray],
    normal_conf: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 5),
    error_max: float = 30.0,
    show: bool = True
) -> plt.Figure:
    """
    可视化置信度对比：左侧真值法向量，中间置信度热力图，右侧角度误差热力图

    Args:
        gt_normal: 真值法向量，支持格式:
            - [3, H, W] (CHW)
            - [H, W, 3] (HWC)
        pred_normal: 预测法向量，格式同gt_normal
        normal_conf: 置信度图，支持格式:
            - [N, 1, H, W] (多帧)
            - [1, H, W] (单帧)
            - [H, W] (单帧)
        mask: 掩码，[H, W]，True表示有效区域
        save_path: 保存路径，如果为None则不保存
        title: 图表标题
        figsize: 图像大小 (width, height)
        error_max: 误差热力图的最大值（度），超过此值的将被截断
        show: 是否调用plt.show()

    Returns:
        matplotlib Figure对象

    使用示例:
        fig = visualize_confidence_comparison(
            gt_normal=gt_normal,
            pred_normal=pred_normal,
            normal_conf=normal_conf,  # [N, 1, H, W] or [H, W]
            mask=mask,
            save_path='confidence_comparison.png'
        )
    """
    # 转换为numpy
    gt = to_numpy(gt_normal)
    pred = to_numpy(pred_normal)
    conf = to_numpy(normal_conf)
    if mask is not None:
        mask = to_numpy(mask)

    # 调整法向量通道顺序：CHW -> HWC
    if gt.ndim == 3 and gt.shape[0] == 3:
        gt = np.transpose(gt, (1, 2, 0))
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))

    # 处理置信度：如果是多帧，取softmax加权平均
    if conf.ndim == 4:  # [N, 1, H, W]
        # 使用softmax权重计算加权平均置信度
        conf_softmax = np.exp(conf) / np.sum(np.exp(conf), axis=0, keepdims=True)
        conf = np.sum(conf * conf_softmax, axis=0)  # [1, H, W]

    if conf.ndim == 3 and conf.shape[0] == 1:  # [1, H, W]
        conf = conf.squeeze(0)  # [H, W]

    # 确保mask是2D
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.squeeze()

    # 计算角度误差
    angle_error = compute_angle_error(gt, pred, mask)

    # 转换为RGB用于可视化
    gt_rgb = normal_to_rgb(gt)

    # 应用mask到RGB图像（背景设为黑色）
    if mask is not None:
        mask_bool = mask.astype(bool)
        gt_rgb[~mask_bool] = 0.0
        conf = np.where(mask_bool, conf, np.nan)

    # 创建图像
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 左侧：真值
    ax_gt = axes[0]
    ax_gt.imshow(gt_rgb)
    ax_gt.set_title('Ground Truth Normal', fontsize=14, fontweight='bold')
    ax_gt.axis('off')

    # 中间：置信度热力图
    ax_conf = axes[1]
    # 使用viridis colormap，但NaN（无效区域）设为黑色
    cmap_conf = plt.get_cmap('viridis')
    cmap_conf.set_bad(color='black')

    # 绘制置信度热力图
    vmin_conf = np.nanmin(conf) if not np.all(np.isnan(conf)) else 0
    vmax_conf = np.nanmax(conf) if not np.all(np.isnan(conf)) else 1
    im_conf = ax_conf.imshow(
        conf,
        cmap=cmap_conf,
        vmin=vmin_conf,
        vmax=vmax_conf
    )
    ax_conf.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax_conf.axis('off')

    # 添加置信度颜色条
    cbar_conf = plt.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar_conf.set_label('Confidence', fontsize=12)

    # 显示置信度统计信息
    if not np.all(np.isnan(conf)):
        mean_conf = np.nanmean(conf)
        std_conf = np.nanstd(conf)
        stats_text = f'Mean: {mean_conf:.2f}\nStd: {std_conf:.2f}'
        ax_conf.text(
            0.02, 0.98, stats_text,
            transform=ax_conf.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

    # 右侧：误差热力图
    ax_error = axes[2]
    # 使用jet colormap，但NaN（无效区域）设为黑色
    cmap_error = plt.get_cmap('jet')
    cmap_error.set_bad(color='black')

    # 绘制热力图
    im_error = ax_error.imshow(
        angle_error,
        cmap=cmap_error,
        vmin=0,
        vmax=error_max
    )
    ax_error.set_title('Angle Error (degrees)', fontsize=14, fontweight='bold')
    ax_error.axis('off')

    # 添加颜色条
    cbar_error = plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
    cbar_error.set_label('Error (°)', fontsize=12)

    # 计算并显示统计信息
    if mask is not None:
        valid_errors = angle_error[mask.astype(bool)]
    else:
        valid_errors = angle_error[~np.isnan(angle_error)]

    if len(valid_errors) > 0:
        mean_err = np.mean(valid_errors)
        median_err = np.median(valid_errors)
        stats_text = f'Mean: {mean_err:.2f}°\nMedian: {median_err:.2f}°'
        ax_error.text(
            0.02, 0.98, stats_text,
            transform=ax_error.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

    # 主标题
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    # 保存
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_batch_confidence_comparison(
    gt_normals: Union[torch.Tensor, np.ndarray],
    pred_normals: Union[torch.Tensor, np.ndarray],
    normal_confs: Union[torch.Tensor, np.ndarray],
    masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_dir: Union[str, Path] = 'vis_results',
    prefix: str = 'conf_sample',
    max_samples: Optional[int] = None,
    **kwargs
) -> None:
    """
    批量可视化置信度对比结果

    Args:
        gt_normals: batch真值法向量 [B, 3, H, W]
        pred_normals: batch预测法向量 [B, 3, H, W]
        normal_confs: batch置信度 [B, N, 1, H, W] 或 [B, 1, H, W]
        masks: batch掩码 [B, H, W]
        save_dir: 保存目录
        prefix: 文件名前缀
        max_samples: 最多可视化多少个样本，None表示全部
        **kwargs: 传递给visualize_confidence_comparison的参数

    使用示例:
        visualize_batch_confidence_comparison(
            gt_normals=batch_gt,
            pred_normals=batch_pred,
            normal_confs=batch_confs,
            masks=batch_masks,
            save_dir='conf_vis_results'
        )
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = gt_normals.shape[0]
    if max_samples is not None:
        batch_size = min(batch_size, max_samples)

    for i in range(batch_size):
        gt = gt_normals[i]
        pred = pred_normals[i]
        conf = normal_confs[i]
        mask = masks[i] if masks is not None else None

        save_path = save_dir / f'{prefix}_{i:03d}.png'

        visualize_confidence_comparison(
            gt_normal=gt,
            pred_normal=pred,
            normal_conf=conf,
            mask=mask,
            save_path=save_path,
            title=f'Confidence Sample {i}',
            show=False,
            **kwargs
        )

    plt.close('all')
    print(f'✓ 批量置信度可视化完成，共保存 {batch_size} 张图片到 {save_dir}')
