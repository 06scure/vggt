"""
光度立体的法向量头 (版本2)

该模块借鉴 DPTHead 的设计，实现表面法向量估计的预测头。
支持处理所有帧的token，输出法向量和置信度。

充分借鉴了 dpt_head.py 的设计模式，包括：
- 分帧处理以节省显存
- 多尺度特征融合
- 置信度预测和激活

注：融合逻辑在外部进行，不在模型内部
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from vggt.heads.utils import create_uv_grid, position_grid_to_embed


class NormalHeadV2(nn.Module):
    """
    法向量预测头V2，基于DPTHead架构实现表面法向量估计。

    该头部借鉴DPTHead的设计，支持处理所有帧的token，
    输出归一化的单位法向量图和置信度图。

    Args:
        dim_in (int): 输入特征维度
        patch_size (int, optional): Patch大小. 默认值为14
        features (int, optional): 中间特征维度. 默认值为256
        out_channels (List[int], optional): 各层输出通道数
        intermediate_layer_idx (List[int], optional): 用于DPT的中间层索引
        pos_embed (bool, optional): 是否使用位置嵌入. 默认值为True
        conf_activation (str, optional): 置信度激活函数. 默认值为"expp1"
        down_ratio (int, optional): 输出下采样比例. 默认值为1
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        conf_activation: str = "expp1",
        down_ratio: int = 1,
    ):
        super().__init__()

        # 基础参数
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.conf_activation = conf_activation
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        # Layer normalization
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0)
             for oc in out_channels]
        )

        # Resize layers for upsampling feature maps
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        # Scratch decoder
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Output layers: 输出4通道 (3个法向量 + 1个置信度)
        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, 4, kernel_size=1, stride=1, padding=0),  # 3通道法向量 + 1通道置信度
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        法向量预测头前向传播，支持处理所有帧的token
        借鉴 DPTHead.forward 的设计

        Args:
            aggregated_tokens_list (List[torch.Tensor]): 聚合后的token列表，来自不同Transformer层
            images (torch.Tensor): 输入图像，形状 [B, N, 3, H, W]，N为光照条件数量
            patch_start_idx (int): Patch token在token序列中的起始索引
            frames_chunk_size (int, optional): 分批处理的帧数量，用于节省显存. 默认值为8

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - normal: 预测的法向量图，形状 [B, N, 3, H, W]，已归一化为单位向量
                - conf: 置信度图，形状 [B, N, 1, H, W]，值越大表示越置信
        """
        B, N, _, H, W = images.shape

        # 如果帧数量小于chunk_size，直接处理所有帧
        if frames_chunk_size is None or frames_chunk_size >= N:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # 否则分批处理以节省显存
        all_normals = []
        all_confs = []

        for frames_start_idx in range(0, N, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, N)

            # 处理当前批次的帧
            chunk_normals, chunk_confs = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_normals.append(chunk_normals)
            all_confs.append(chunk_confs)

        # 沿着序列维度拼接结果
        normal = torch.cat(all_normals, dim=1)
        conf = torch.cat(all_confs, dim=1)

        return normal, conf

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播的具体实现，支持处理特定范围的帧

        Args:
            aggregated_tokens_list (List[torch.Tensor]): 聚合后的token列表
            images (torch.Tensor): 输入图像
            patch_start_idx (int): Patch token起始索引
            frames_start_idx (int, optional): 起始帧索引
            frames_end_idx (int, optional): 结束帧索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (法向量, 置信度)
        """
        # 如果指定了帧范围，选择对应的图像
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        # 从每个中间层提取所有帧的特征
        for layer_idx in self.intermediate_layer_idx:
            # 形状: [B, S, T, C] -> 取所有帧
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # 如果指定了帧范围，选择对应的token
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            # reshape: [B, S, T, C] -> [B*S, T, C]
            x = x.reshape(B * S, -1, x.shape[-1])

            # 归一化
            x = self.norm(x)

            # reshape为特征图 [B*S, C, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape((B * S, x.shape[-1], patch_h, patch_w))

            # 投影和缩放
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # 多尺度特征融合
        out = self._scratch_forward(out)

        # 恢复到目标分辨率
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # 输出层: [B*S, 4, H, W]
        out = self.scratch.output_conv2(out)

        # 分离法向量和置信度（借鉴 activate_head 的方式）
        # out: [B*S, 4, H, W] -> permute -> [B*S, H, W, 4]
        fmap = out.permute(0, 2, 3, 1)

        # 分离: normal [B*S, H, W, 3], conf [B*S, H, W, 1]
        normal = fmap[:, :, :, :3]
        conf = fmap[:, :, :, 3:]

        # 归一化法向量到单位长度
        normal = F.normalize(normal, p=2, dim=-1)

        # 置信度激活
        if self.conf_activation == "expp1":
            conf = 1 + torch.exp(conf)
        elif self.conf_activation == "expp0":
            conf = torch.exp(conf)
        elif self.conf_activation == "sigmoid":
            conf = torch.sigmoid(conf)
        else:
            raise ValueError(f"Unknown conf_activation: {self.conf_activation}")

        # permute back: [B*S, H, W, 3] -> [B*S, 3, H, W]
        normal = normal.permute(0, 3, 1, 2)
        conf = conf.permute(0, 3, 1, 2)

        # reshape回 [B, S, 3, H, W] 和 [B, S, 1, H, W]
        normal = normal.view(B, S, 3, H, W)
        conf = conf.view(B, S, 1, H, W)

        return normal, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        应用位置嵌入
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        多尺度特征融合

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.        
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        out = self.scratch.refinenet1(out, layer_1_rn)
        out = self.scratch.output_conv1(out)
        return out


def fuse_normals_with_confidence(
    normals: torch.Tensor,
    confidences: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    使用置信度加权融合多帧法向量预测

    Args:
        normals: 多帧法向量预测，形状 [B, N, 3, H, W]
        confidences: 置信度图，形状 [B, N, 1, H, W]
        mask: 可选的mask，形状 [B, 1, H, W] 或 [B, H, W]

    Returns:
        融合后的法向量，形状 [B, 3, H, W]
    """
    B, N, _, H, W = normals.shape

    # 使用softmax将置信度转换为权重
    # confidences: [B, N, 1, H, W] -> weights: [B, N, 1, H, W]
    weights = torch.softmax(confidences, dim=1)

    # 加权融合法向量
    # normals: [B, N, 3, H, W], weights: [B, N, 1, H, W]
    fused_normal = torch.sum(normals * weights, dim=1)  # [B, 3, H, W]

    # 重新归一化到单位长度
    fused_normal = F.normalize(fused_normal, p=2, dim=1)

    # 应用mask（如果提供）
    if mask is not None:
        if mask.dim() == 3:  # [B, H, W]
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        fused_normal = fused_normal * mask

    return fused_normal


################################################################################
# Modules 
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init."""
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass."""
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init."""
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass."""
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: tuple = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
