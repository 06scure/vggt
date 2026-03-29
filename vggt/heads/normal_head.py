"""
光度立体的法向量头

该模块实现了表面法向量估计的预测头。
基于 DPT（密集预测 Transformer）架构，
专门用于预测单位表面法向量，只使用第0帧token。

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from vggt.heads.utils import create_uv_grid, position_grid_to_embed


class NormalHead(nn.Module):
    """
    法向量预测头，基于DPT架构实现表面法向量估计。

    该头部专门用于光度立体任务，接收聚合后的特征并输出
    归一化的单位法向量图。只使用第0帧的token进行预测。

    Args:
        dim_in (int): 输入特征维度
        patch_size (int, optional): Patch大小. 默认值为14
        features (int, optional): 中间特征维度. 默认值为256
        out_channels (List[int], optional): 各层输出通道数
        intermediate_layer_idx (List[int], optional): 用于DPT的中间层索引
        pos_embed (bool, optional): 是否使用位置嵌入. 默认值为True
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
    ):
        super().__init__()

        # 基础参数
        self.patch_size = patch_size
        self.pos_embed = pos_embed
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

        # Output layers
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, 3, kernel_size=1, stride=1, padding=0),  # 3通道法向量
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        """
        法向量预测头前向传播

        Args:
            aggregated_tokens_list (List[torch.Tensor]): 聚合后的token列表，来自不同Transformer层
            images (torch.Tensor): 输入图像，形状 [B, N, 3, H, W]，N为光照条件数量
            patch_start_idx (int): Patch token在token序列中的起始索引

        Returns:
            torch.Tensor: 预测的法向量图，形状 [B, 3, H, W]，已归一化为单位向量
        """
        B, _, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        # 从每个中间层提取第0帧的特征
        for layer_idx in self.intermediate_layer_idx:
            # 形状: [B, S, T, C] -> 取第0帧: [B, T, C]
            x = aggregated_tokens_list[layer_idx][:, 0, patch_start_idx:]

            # 归一化
            x = self.norm(x)

            # reshape为特征图 [B, C, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape((B, x.shape[-1], patch_h, patch_w))

            # 投影和缩放
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # 多尺度特征融合
        out = self._scratch_forward(out)

        # 恢复到原始图像分辨率
        # 特征融合后的尺寸小于原始尺寸，必须使用插值恢复以保持像素位置对应关系
        out = F.interpolate(
            out,
            (patch_h * self.patch_size, patch_w * self.patch_size),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # 输出层
        out = self.scratch.output_conv2(out)

        # 归一化到单位向量
        out = F.normalize(out, p=2, dim=1)

        return out

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """应用位置嵌入"""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """多尺度特征融合"""
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
        """Init.

        Args:
            features (int): number of features
        """
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
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

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
        """Init.

        Args:
            features (int): number of features
        """
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
        """Forward pass.

        Returns:
            tensor: output
        """
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
    size: Tuple[int, int] = None,
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
