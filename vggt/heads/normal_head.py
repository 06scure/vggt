#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpt_head import DPTHead


class NormalHead(nn.Module):
    """
    基于DPT架构的法向量预测头

    输入: Aggregator输出的特征
    输出: 3通道法向量图，归一化到单位长度
    使用第0帧的特征frame_attention进行预测
    """
    def __init__(self, dim_in: int, patch_size: int = 14, output_dim: int = 3, activation: str = "linear", conf_activation: str = "expp1"):
        super().__init__()
        self.patch_size = patch_size
        self.dpt_head = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim + 1,  # 3个法向量分量 + 1个置信度
            activation=activation,
            conf_activation=conf_activation,
            feature_only=False
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx):
        """
        前向传播

        Args:
            aggregated_tokens_list: 聚合后的token列表
            images: 输入图像 [B, S, 3, H, W]
            patch_start_idx: patch token起始索引

        Returns:
            法向量图 [B, 3, H, W]，已归一化到单位长度
        """
        # 使用DPTHead进行预测
        preds, conf = self.dpt_head(aggregated_tokens_list, images, patch_start_idx)

        # preds shape: [B, S, H, W, 3]
        # 只取第0帧的预测结果
        preds = preds[:, 0]  # [B, H, W, 3]

        # 转换为 [B, 3, H, W]
        preds = preds.permute(0, 3, 1, 2)

        # 归一化到单位长度
        preds = F.normalize(preds, p=2, dim=1, eps=1e-6)

        return preds, conf