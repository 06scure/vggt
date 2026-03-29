"""
光度立体任务的损失函数模块 (版本2)

提供法向量预测的损失计算，包括：
- MSE损失（带掩码）
- 余弦相似度损失（带掩码）
- 不确定性损失（借鉴VGGT的aleatoric uncertainty loss）

支持对每一帧的法向量预测进行监督，并用置信度加权。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional

from training.train_utils.general import check_and_fix_inf_nan


@dataclass(eq=False)
class PSLossV2(nn.Module):
    """
    光度立体任务的损失模块V2

    支持：
    - 多帧法向量MSE损失（带不确定性加权）
    - 多帧法向量余弦相似度损失（带不确定性加权）
    - 不确定性正则化损失
    """
    def __init__(
        self,
        normal=None,
        uncertainty_weight: float = 0.1,
        use_fused_loss: bool = False,
        use_per_frame_loss: bool = True,
        **kwargs
    ):
        super().__init__()
        # 法向量损失配置
        self.normal = normal if normal is not None else {"weight": 1.0, "loss_type": "mse"}
        # 不确定性损失的权重
        self.uncertainty_weight = uncertainty_weight
        # 是否使用融合后的法向量计算损失
        self.use_fused_loss = use_fused_loss
        # 是否使用每一帧的法向量计算损失
        self.use_per_frame_loss = use_per_frame_loss

    def forward(self, predictions, batch) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            predictions: 包含模型预测的字典，需要包含：
                - 'normal_all': [B, N, 3, H, W] 所有帧的法向量预测
                - 'normal_conf': [B, N, 1, H, W] 所有帧的置信度
                - 'normal': [B, 3, H, W] 融合后的法向量（可选）
            batch: 包含ground truth的字典，需要包含 'normal' 和 'mask'

        Returns:
            包含各损失项和总目标的字典
        """
        total_loss = 0
        loss_dict = {}

        # 法向量损失
        if "normal_all" in predictions:
            normal_loss_dict = compute_normal_loss_v2(
                predictions,
                batch,
                uncertainty_weight=self.uncertainty_weight,
                use_fused_loss=self.use_fused_loss,
                use_per_frame_loss=self.use_per_frame_loss,
                **self.normal
            )
            normal_loss = normal_loss_dict["loss_normal"] * self.normal["weight"]
            total_loss = total_loss + normal_loss
            loss_dict.update(normal_loss_dict)

        loss_dict["objective"] = total_loss
        return loss_dict


def compute_normal_loss_v2(
    pred_dict: Dict[str, torch.Tensor],
    batch_data: Dict[str, torch.Tensor],
    loss_type: str = "mse",
    uncertainty_weight: float = 0.1,
    use_fused_loss: bool = True,
    use_per_frame_loss: bool = True,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    计算法向量预测损失V2 - 支持不确定性损失

    借鉴VGGT的aleatoric uncertainty loss：
    L = Σ [ (1/σ²) * ||pred - gt||² + α log σ ]

    Args:
        pred_dict: 预测字典，包含：
            - 'normal_all': [B, N, 3, H, W] 所有帧的法向量预测
            - 'normal_conf': [B, N, 1, H, W] 所有帧的置信度（σ = conf）
            - 'normal': [B, 3, H, W] 融合后的法向量（可选）
        batch_data: ground truth字典，包含：
            - 'normal': [B, 3, H, W] 或 [3, H, W]，单位法向量
            - 'mask': [B, H, W] 或 [H, W]，布尔掩码，True表示有效区域
        loss_type: 损失类型，'mse' 或 'cosine'
        uncertainty_weight: 不确定性正则化项的权重 α
        use_fused_loss: 是否使用融合后的法向量计算损失
        use_per_frame_loss: 是否使用每一帧的法向量计算损失
        **kwargs: 其他参数

    Returns:
        包含损失项的字典
    """
    # 获取所有帧的预测
    pred_normal_all = pred_dict["normal_all"]  # [B, N, 3, H, W]
    pred_conf_all = pred_dict["normal_conf"]    # [B, N, 1, H, W]

    B, N, _, H, W = pred_normal_all.shape

    # 获取ground truth法向量
    gt_normal = batch_data["normal"]
    if len(gt_normal.shape) == 3:  # [3, H, W] -> [1, 3, H, W]
        gt_normal = gt_normal.unsqueeze(0)

    # 确保GT法向量和预测法向量有相同的batch size
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
    valid_mask = mask.to(torch.bool)  # [B, H, W]
    valid_count = torch.sum(valid_mask)

    # 如果有效像素太少，返回0损失
    if valid_count < 10:
        return {
            "loss_normal": torch.tensor(0.0, device=pred_normal_all.device),
            "loss_normal_data": torch.tensor(0.0, device=pred_normal_all.device),
            "loss_normal_uncertainty": torch.tensor(0.0, device=pred_normal_all.device),
        }

    loss_dict = {}
    total_loss = 0.0

    # 将GT扩展到所有帧: [B, 3, H, W] -> [B, N, 3, H, W]
    gt_normal_all = gt_normal.unsqueeze(1).expand(B, N, 3, H, W)

    # 将mask扩展到所有帧: [B, H, W] -> [B, N, H, W]
    valid_mask_all = valid_mask.unsqueeze(1).expand(B, N, H, W)

    # ==========================
    # 每一帧的损失（带不确定性）
    # ==========================
    if use_per_frame_loss:
        if loss_type == "mse":
            # 计算每帧的MSE误差: [B, N, 3, H, W] -> [B, N, H, W]
            error = F.mse_loss(pred_normal_all, gt_normal_all, reduction="none")
            error = error.mean(dim=2)  # 在通道维度平均
        elif loss_type == "cosine":
            # 计算余弦相似度损失: 1 - cos(theta)
            pred_normal_normalized = F.normalize(pred_normal_all, p=2, dim=2)
            dot_product = torch.sum(pred_normal_normalized * gt_normal_all, dim=2)  # [B, N, H, W]
            dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
            error = 1.0 - dot_product  # [B, N, H, W]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # 置信度作为不确定性 σ = conf
        sigma = pred_conf_all.squeeze(2)  # [B, N, 1, H, W] -> [B, N, H, W]

        # 数据项: (1/σ²) * error
        # 注意：sigma使用expp1激活，sigma >= 1，所以1/sigma²不会爆炸
        data_loss = error / (sigma ** 2 + 1e-8)  # [B, N, H, W]

        # 不确定性正则项: α * log(σ)
        # 鼓励模型在不确定的地方预测更大的sigma，但不要太大
        uncertainty_loss = torch.log(sigma + 1e-8)  # [B, N, H, W]

        # 应用mask
        data_loss = data_loss[valid_mask_all]
        uncertainty_loss = uncertainty_loss[valid_mask_all]

        # 计算平均值
        data_loss = check_and_fix_inf_nan(data_loss, "normal_loss_data")
        uncertainty_loss = check_and_fix_inf_nan(uncertainty_loss, "normal_loss_uncertainty")

        data_loss = data_loss.mean()
        uncertainty_loss = uncertainty_loss.mean()

        # 总损失 = 数据项 + α * 不确定性项
        per_frame_loss = data_loss + uncertainty_weight * uncertainty_loss

        loss_dict["loss_normal_data"] = data_loss
        loss_dict["loss_normal_uncertainty"] = uncertainty_loss
        loss_dict["loss_normal_per_frame"] = per_frame_loss

        total_loss = total_loss + per_frame_loss

    # ==========================
    # 融合后法向量的损失
    # ==========================
    if use_fused_loss and "normal" in pred_dict:
        pred_normal_fused = pred_dict["normal"]  # [B, 3, H, W]

        if loss_type == "mse":
            fused_loss = F.mse_loss(pred_normal_fused, gt_normal, reduction="none")
            fused_loss = fused_loss.mean(dim=1)  # [B, H, W]
        elif loss_type == "cosine":
            pred_normal_normalized = F.normalize(pred_normal_fused, p=2, dim=1)
            dot_product = torch.sum(pred_normal_normalized * gt_normal, dim=1)
            dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
            fused_loss = 1.0 - dot_product  # [B, H, W]

        fused_loss = fused_loss[valid_mask]
        fused_loss = check_and_fix_inf_nan(fused_loss, "normal_loss_fused")
        fused_loss = fused_loss.mean()

        loss_dict["loss_normal_fused"] = fused_loss

        # 融合损失的权重设为0.5，避免与per-frame loss冲突
        total_loss = total_loss + 0.5 * fused_loss

    loss_dict["loss_normal"] = total_loss

    return loss_dict


def normal_mse_loss_v2(
    prediction_all: torch.Tensor,
    conf_all: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    uncertainty_weight: float = 0.1,
) -> torch.Tensor:
    """
    简单的法向量MSE损失函数V2（带不确定性）

    Args:
        prediction_all: 所有帧预测的法向量 [B, N, 3, H, W]
        conf_all: 所有帧的置信度 [B, N, 1, H, W]
        target: 真实的法向量 [B, 3, H, W]
        mask: 可选掩码 [B, H, W]，True表示有效区域
        uncertainty_weight: 不确定性正则项的权重

    Returns:
        损失值
    """
    B, N, _, H, W = prediction_all.shape

    # 将target扩展到所有帧
    target_all = target.unsqueeze(1).expand(B, N, 3, H, W)

    # 计算MSE误差
    error = F.mse_loss(prediction_all, target_all, reduction="none")
    error = error.mean(dim=2)  # [B, N, H, W]

    # 置信度作为不确定性
    sigma = conf_all.squeeze(2)  # [B, N, H, W]

    # 数据项 + 不确定性正则项
    data_loss = error / (sigma ** 2 + 1e-8)
    uncertainty_loss = torch.log(sigma + 1e-8)

    loss = data_loss + uncertainty_weight * uncertainty_loss

    # 应用掩码
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask_all = mask.unsqueeze(1).expand(B, N, H, W)
        valid_mask = mask_all.to(torch.bool)
        valid_count = torch.sum(valid_mask)

        if valid_count > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean()
    else:
        loss = loss.mean()

    return check_and_fix_inf_nan(loss, "normal_mse_loss_v2")
