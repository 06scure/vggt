import torch
import torch.nn as nn
import torch.nn.functional as F


class PSLoss(nn.Module):
    """
    光度立体法向量预测损失函数

    计算预测法向量与真实法向量之间的MSE损失，
    只计算mask内的有效区域，忽略背景信息。
    """
    def __init__(self, mask_weight=1.0):
        """
        初始化PS损失函数

        Args:
            mask_weight: mask的权重，默认为1.0
        """
        super().__init__()
        self.mask_weight = mask_weight
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, predictions, batch):
        """
        计算法向量预测损失

        Args:
            predictions: 模型预测结果，包含 'normals' 键
            batch: 真实数据，包含 'gt_normal' 和 'mask' 键

        Returns:
            损失值字典
        """
        loss_dict = {}

        # 检查是否有法向量预测
        if 'normals' not in predictions:
            return loss_dict

        # 获取预测法向量和真实法向量
        pred_normals = predictions['normals']
        gt_normals = batch['gt_normal']

        # 确保预测和真实值形状一致
        if pred_normals.shape != gt_normals.shape:
            pred_normals = F.interpolate(
                pred_normals,
                size=gt_normals.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        # 获取mask
        mask = batch['mask']
        if mask is not None:
            # 确保mask形状与法向量一致
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[-2:] != pred_normals.shape[-2:]:
                mask = F.interpolate(
                    mask.float(),
                    size=pred_normals.shape[-2:],
                    mode='nearest'
                ).bool()

            # 只计算mask内的损失
            valid_mask = mask.squeeze(1)
            pred_valid = pred_normals.permute(0, 2, 3, 1)[valid_mask]
            gt_valid = gt_normals.permute(0, 2, 3, 1)[valid_mask]
        else:
            # 没有mask时计算整个图像的损失
            pred_valid = pred_normals.permute(0, 2, 3, 1).reshape(-1, 3)
            gt_valid = gt_normals.permute(0, 2, 3, 1).reshape(-1, 3)

        # 计算MSE损失
        if pred_valid.numel() > 0:
            loss = self.mse_loss(pred_valid, gt_valid) / (pred_valid.size(0) + 1e-8)
        else:
            loss = torch.tensor(0.0, device=pred_normals.device)

        loss_dict['loss_normal'] = loss

        return loss_dict


def compute_ps_loss(predictions, batch, loss_config):
    """
    计算光度立体任务的损失

    Args:
        predictions: 模型预测结果
        batch: 真实数据
        loss_config: 损失配置字典

    Returns:
        损失值字典
    """
    loss_fn = PSLoss(**loss_config)
    return loss_fn(predictions, batch)