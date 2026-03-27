"""
DiLiGenT 数据集类 - 用于光度立体任务的标准测试集

数据集特点：
- 包含10个真实物体，每个物体96张不同光照的图像
- 提供高质量的ground truth法向量图和物体mask
- 原始图像分辨率：518×518
- 支持随机采样两种模式
- 无数据增强，保持像素位置一致性
"""
import torch
import logging
import numpy as np
from typing import Union

from training.data.datasets.ps_dataset import BasePSDataset


logger = logging.getLogger(__name__)


class DiLiGenTDataset(BasePSDataset):
    """
    DiLiGenT测试数据集
    数据路径：/home/user/dataset/DiLiGenT_518
    """

    def __init__(
        self,
        data_dir: str = "/home/user/dataset/DiLiGenT_518",
        img_size: int = 518, 
        img_per_seq: int = 10,
        split: str = "test",
    ):
        """
        初始化DiLiGenT数据集

        Args:
            data_dir: 数据集根目录
            img_size: 目标图像尺寸，默认518
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
        每个数据序列是一个包含不同光照图像、mask.png和normal.png的文件夹

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

            img_paths = sorted(seq_dir.glob("*.png"))
            mask_path = seq_dir / "mask.png"
            normal_path = seq_dir / "Normal_gt.png"

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

    def _normalize_gt(self, gt):
        """
        归一化DiLiGenT数据集的法向量为单位向量

        DiLiGenT数据集的法向量存储在PNG图像中,其向量普遍偏大,模长约等于1.7
        Args:
            gt: 法向量张量 [3, H, W]
        Returns:
            归一化后的法向量张量 [3, H, W]
        """
        # 计算每个像素的法向量模长
        norm = torch.norm(gt, dim=0, keepdim=True)
        # 避免除以0，只对模长大于1e-6的像素进行归一化
        normalized_gt = torch.where(
            norm > 1e-6,
            gt / norm,
            gt  # 对于模长很小的像素保持原值
        )
        return normalized_gt