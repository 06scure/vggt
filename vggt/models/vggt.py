# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from huggingface_hub import constants

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.normal_head import NormalHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_normal=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.normal_head = NormalHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_normal else None

    def freeze_aggregator(self):
        """冻结 aggregator 的参数，只训练 heads"""
        for param in self.aggregator.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - normals (torch.Tensor): Predicted surface normal maps with shape [B, 3, H, W] (only frame 0)
                - normals_conf (torch.Tensor): Confidence scores for normal predictions with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.normal_head is not None:
                normals, normals_conf = self.normal_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["normals"] = normals
                predictions["normals_conf"] = normals_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions


    @classmethod
    def _from_pretrained(cls, *, model_id: str, revision: Optional[str],
                    cache_dir: Optional[Union[str, Path]], force_download: bool,
                    local_files_only: bool, token: Union[str, bool, None],
                    map_location: str = "cpu", strict: bool = False,
                    **model_kwargs):
        """重写方法，修复本地加载时的格式检测问题"""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")

            # 尝试加载 safetensors 格式
            safetensors_path = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            if os.path.exists(safetensors_path):
                # 使用父类的 _load_as_safetensor 方法
                return PyTorchModelHubMixin._load_as_safetensor(model, safetensors_path, map_location, strict)

            # 尝试加载 PyTorch 格式 (支持 pytorch_model.bin 和 model.pt)
            pytorch_path1 = os.path.join(model_id, constants.PYTORCH_WEIGHTS_NAME)  # pytorch_model.bin
            pytorch_path2 = os.path.join(model_id, "model.pt")  # 你的文件格式

            if os.path.exists(pytorch_path1):
                return PyTorchModelHubMixin._load_as_pickle(model, pytorch_path1, map_location, strict)
            elif os.path.exists(pytorch_path2):
                return PyTorchModelHubMixin._load_as_pickle(model, pytorch_path2, map_location, strict)

            raise FileNotFoundError(f"No model weights found in {model_id}. "
                                  f"Expected files: {constants.SAFETENSORS_SINGLE_FILE}, "
                                  f"{constants.PYTORCH_WEIGHTS_NAME}, or model.pt")
        else:
            return super()._from_pretrained(
                model_id=model_id, revision=revision, cache_dir=cache_dir,
                force_download=force_download, local_files_only=local_files_only,
                token=token, map_location=map_location, strict=strict,
                **model_kwargs
            )