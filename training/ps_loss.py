"""
光度立体任务的损失函数模块

提供法向量预测的损失计算，包括：
- MSE损失（带掩码）
- 余弦相似度损失（带掩码）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional

from training.train_utils.general import check_and_fix_inf_nan


@dataclass(eq=False)
class PSLoss(nn.Module):
    """
    光度立体任务的损失模块

    支持：
    - 法向量MSE损失
    - 法向量余弦相似度损失
    """
    def __init__(self, normal=None, **kwargs):
        super().__init__()
        # 法向量损失配置
        self.normal = normal if normal is not None else {"weight": 1.0, "loss_type": "mse"}

    def forward(self, predictions, batch) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            predictions: 包含模型预测的字典，需要包含 'normal'
            batch: 包含ground truth的字典，需要包含 'normal' 和 'mask'

        Returns:
            包含各损失项和总目标的字典
        """
        total_loss = 0
        loss_dict = {}

        # 法向量损失
        if "normal" in predictions:
            normal_loss_dict = compute_normal_loss(
                predictions, batch, **self.normal
            )
            normal_loss = normal_loss_dict["loss_normal"] * self.normal["weight"]
            total_loss = total_loss + normal_loss
            loss_dict.update(normal_loss_dict)

        loss_dict["objective"] = total_loss
        return loss_dict


def compute_normal_loss(
    pred_dict: Dict[str, torch.Tensor],
    batch_data: Dict[str, torch.Tensor],
    loss_type: str = "mse",
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    计算法向量预测损失

    Args:
        pred_dict: 预测字典，包含 'normal' [B, 3, H, W]
        batch_data: ground truth字典，包含：
            - 'normal': [B, 3, H, W] 或 [3, H, W]，单位法向量
            - 'mask': [B, H, W] 或 [H, W]，布尔掩码，True表示有效区域
        loss_type: 损失类型，'mse' 或 'cosine'
        **kwargs: 其他参数

    Returns:
        包含损失项的字典
    """
    # 获取预测的法向量
    pred_normal = pred_dict["normal"]  # [B, 3, H, W]

    # 获取ground truth法向量
    gt_normal = batch_data["normal"]
    if len(gt_normal.shape) == 3:  # [3, H, W] -> [1, 3, H, W]
        gt_normal = gt_normal.unsqueeze(0)

    # 确保GT法向量和预测法向量有相同的batch size
    B = pred_normal.shape[0]
    if gt_normal.shape[0] == 1 and B > 1:
        gt_normal = gt_normal.expand(B, -1, -1, -1)

    # 获取掩码
    mask = batch_data["mask"]
    if len(mask.shape) == 2:  # [H, W] -> [1, H, W]
        mask = mask.unsqueeze(0)

    # 确保掩码和法向量有相同的batch size
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1)

    # 只计算有效区域的损失
    valid_mask = mask.to(torch.bool)
    valid_count = torch.sum(valid_mask)

    # 如果有效像素太少，返回0损失
    if valid_count < 10:
        return {
            "loss_normal": torch.tensor(0.0, device=pred_normal.device),
            "loss_normal_mse": torch.tensor(0.0, device=pred_normal.device) if loss_type == "mse" else None,
            "loss_normal_cosine": torch.tensor(0.0, device=pred_normal.device) if loss_type == "cosine" else None,
        }

    loss_dict = {}

    if loss_type == "mse":
        # MSE损失：比较法向量的每个通道
        loss = F.mse_loss(pred_normal, gt_normal, reduction="none")  # [B, 3, H, W]
        # 在通道维度求平均，然后应用掩码
        loss = loss.mean(dim=1)  # [B, H, W]
        loss = loss[valid_mask]
        loss = check_and_fix_inf_nan(loss, "normal_loss_mse")
        loss = loss.mean()
        loss_dict["loss_normal"] = loss
        loss_dict["loss_normal_mse"] = loss

    elif loss_type == "cosine":
        # 余弦相似度损失：1 - cos(theta)，其中theta是预测法向量和GT法向量的夹角
        # 归一化预测法向量（确保是单位向量）
        pred_normal_normalized = F.normalize(pred_normal, p=2, dim=1)
        # 计算点积（余弦相似度）
        dot_product = torch.sum(pred_normal_normalized * gt_normal, dim=1)  # [B, H, W]
        # 限制在[-1, 1]范围内
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        # 损失 = 1 - cos(theta)
        loss = 1.0 - dot_product
        loss = loss[valid_mask]
        loss = check_and_fix_inf_nan(loss, "normal_loss_cosine")
        loss = loss.mean()
        loss_dict["loss_normal"] = loss
        loss_dict["loss_normal_cosine"] = loss

    else:
        raise ValueError(f"Unknown loss type: {loss_type}, choose 'mse' or 'cosine'")

    return loss_dict


def normal_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    简单的法向量MSE损失函数（带掩码）

    Args:
        prediction: 预测的法向量 [B, 3, H, W] 或 [3, H, W]
        target: 真实的法向量 [B, 3, H, W] 或 [3, H, W]
        mask: 可选掩码 [B, H, W] 或 [H, W]，True表示有效区域

    Returns:
        损失值
    """
    # 确保有batch维度
    if len(prediction.shape) == 3:
        prediction = prediction.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)

    # 计算MSE损失
    loss = F.mse_loss(prediction, target, reduction="none")  # [B, 3, H, W]
    loss = loss.mean(dim=1)  # [B, H, W]

    # 应用掩码
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        # 确保mask和loss有相同的batch size
        if mask.shape[0] == 1 and prediction.shape[0] > 1:
            mask = mask.expand(prediction.shape[0], -1, -1)

        valid_mask = mask.to(torch.bool)
        valid_count = torch.sum(valid_mask)

        if valid_count > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean()
    else:
        loss = loss.mean()

    return check_and_fix_inf_nan(loss, "normal_mse_loss")