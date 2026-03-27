import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class BasePSDataset(Dataset):
    """
    光度立体任务数据集抽象基类
    
    提供通用的加载、裁剪、归一化逻辑
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        img_size: int,
        img_per_seq: int,
        split: str = "train",
    ):
        """
        初始化基类数据集
        
        Args:
            data_dir: 数据集根目录
            img_size: 目标图像尺寸（必须是14的倍数，适配DINOv2）
            img_per_seq: 每个序列使用的图像数量
            split: 数据集分割，'train' 或 'test'
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.img_per_seq = img_per_seq
        self.split = split
        
        # 验证图像尺寸是14的倍数
        assert img_size % 14 == 0, f"Image size {img_size} must be divisible by 14 for DINOv2"

        # 收集所有有效元素
        self.item = self._collect_item()
        logger.info(f"Loaded {len(self.item)} item from {data_dir}")
        
    def _collect_item(self) -> list:
        """
        收集所有有效元素(子类必须实现）
        
        Returns:
            list: 序列列表，每个元素代表一个有效序列
        """
        raise NotImplementedError("_collect_item must be implemented by subclass")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.item)
    
    def _load_image (self, path: Union[str, Path]) -> torch.Tensor:
        """
        加载并预处理图像
        
        Args:
            path: 图像路径
            
        Returns:
            预处理后的图像张量 [3, H, W]，范围 [0, 1]
        """
        raise NotImplementedError("_load_image must be implemented by subclass")
    
    def _load_mask (self, path: Union[str, Path]) -> torch.Tensor:
        """
        加载并预处理mask图像
        
        Args:
            path: 图像路径
            
        Returns:
            预处理后的图像张量 [H, W]，范围 [0, 1]
        """
        raise NotImplementedError("_load_mask must be implemented by subclass")
    
    def _load_normal(self, path: Union[str, Path]) -> torch.Tensor:
        """
        加载并预处理法向量图
        
        Args:
            path: 法向量图路径
            
        Returns:
            预处理后的法向量张量 [3, H, W]，单位向量，范围 [-1, 1]
        """
        raise NotImplementedError("_load_normal must be implemented by subclass")
    
    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        """
        获取数据集中的一个样本（子类必须实现）
        
        Args:
            idx: 样本索引
            
        Returns:
            包含数据的字典，至少包含'images', 'normals', 'masks'
        """
        raise NotImplementedError("__getitem__ must be implemented by subclass")

# from dataset_util.py
def read_image_cv2(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img