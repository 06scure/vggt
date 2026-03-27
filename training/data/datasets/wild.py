"""
PSWild 数据集类 - 用于光度立体任务的真实世界物体图像训练集

数据集特点：
- 均为合成数据集，在不同光照条件下的软件渲染图像
- 每个物体包含10张不同光照的图像
- 提供ground truth法向量图和物体 mask
- 原始图像分辨率：512×512 → 中心裁剪到 504×504（14的倍数）
- 无数据增强，保持像素位置一致性
"""
import torch
import logging
import numpy as np
from typing import Union

from training.data.datasets.ps_dataset import BasePSDataset


logger = logging.getLogger(__name__)

class PSWildDataset(BasePSDataset):
    """
    PSWild训练数据集
    数据路径：/home/user/dataset/PSWild
    """

    def __init__(
        self,
        data_dir: str = "/home/user/dataset/PSWild",
        img_size: int = 504,  # 512中心裁剪到504，是14的倍数
        img_per_seq: int = 10,
        split: str = "train",
    ):
        """
        初始化PSWild数据集
        
        Args:
            data_dir: 数据集根目录
            img_size: 目标图像尺寸，默认504（必须是14的倍数）
            img_per_seq: 每个序列使用的图像数量，默认10
            split: 数据集分割，'train' 或 'test'
        """
        super().__init__(
            data_dir=data_dir,
            img_size=img_size,
            img_per_seq=img_per_seq,
            split=split
        )
    
    def _collect_item(self):
        """
        收集所有有效的数据序列
        每个数据序列是一个包含不同光照图像、mask.png和normal.tif的文件夹

        Returns:
            list[Path]: 序列目录路径列表
        """
        items = []
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return items

        for seq_dir in self.data_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            img_paths = sorted(seq_dir.glob("00*.tif"))
            mask_path = seq_dir / "mask.png"
            normal_path = seq_dir / "normal.tif"

            if len(img_paths) < self.img_per_seq:
                logger.warning(f"Sequence {seq_dir} has fewer than {self.img_per_seq} images, skipping")
                continue

            if not mask_path.exists() or not normal_path.exists():
                logger.warning(f"Sequence {seq_dir} is missing mask or normal files, skipping")
                continue

            items.append({
                "img_paths": img_paths, 
                "mask_path": mask_path,
                "normal_path": normal_path
            })

        return items

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        """
        获取数据集中的一个样本

        Args:
            idx: 样本索引

        Returns:
            包含数据的字典，至少包含'images', 'normal', 'mask'
        """
        item = self.item[idx]

        # 随机采样 img_per_seq 张图像
        img_paths = item["img_paths"]
        if len(img_paths) > self.img_per_seq:
            # 随机选择 img_per_seq 张图像
            selected_indices = np.random.choice(len(img_paths), self.img_per_seq, replace=False)
            selected_paths = [img_paths[i] for i in selected_indices]
        else:
            # 如果图像数量不足，直接使用所有图像
            selected_paths = img_paths

        # 加载图像
        images = []
        for img_path in selected_paths:
            img = self._load_image(img_path)
            images.append(img)

        # 加载mask
        mask = self._load_mask(item["mask_path"])

        # 加载normal
        normal = self._load_normal(item["normal_path"])

        # 转换为torch.Tensor
        images_tensor = self._convert_dtype(images)

        return {
            "idx": idx,
            "images": images_tensor,
            "normal": normal,
            "mask": mask
        }

    def _crop_to_target_size(self, img: np.ndarray) -> np.ndarray:
        """
        中心裁剪图像到目标尺寸

        Args:
            img: 输入图像数组 [H, W, C] 或 [H, W]

        Returns:
            裁剪后的图像数组 [img_size, img_size, C] 或 [img_size, img_size]
        """
        # wild 数据集的原始size是512x512，需要裁剪到504x504（14的倍数）
        h, w = img.shape[:2]
        target_size = self.img_size

        # 计算裁剪偏移量
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        bottom = top + target_size
        right = left + target_size

        if len(img.shape) == 2:
            # 灰度图像
            return img[top:bottom, left:right]
        else:
            return img[top:bottom, left:right, :]