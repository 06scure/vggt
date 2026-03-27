import os
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
        split: str,
        rgb = True,
    ):
        """
        初始化基类数据集
        
        Args:
            data_dir: 数据集根目录
            img_size: 目标图像尺寸（必须是14的倍数，适配DINOv2）
            img_per_seq: 每个序列使用的图像数量
            split: 数据集分割，'train' 或 'test'
            rgb: 是否将图像转换为RGB格式
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.img_per_seq = img_per_seq
        self.split = split
        self.rgb = rgb
        
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
    
    def _load_image (self, path: Union[str, Path]) -> np.ndarray:
        """
        加载并预处理图像

        Args:
            path: 图像路径

        Returns:
            预处理后的图像数组 [H, W, 3]
        """
        assert os.path.exists(path) and os.path.getsize(path) > 0, f"File does not exist or is empty: {path}"
        img = read_image_cv2(str(path), rgb=self.rgb)   # 把图像加载为RGB格式
        img = self._crop_to_target_size(img)  # 裁剪到目标尺寸

        return img

    def _load_mask (self, path: Union[str, Path]) -> torch.Tensor:
        """
        加载并预处理mask图像

        Args:
            path: 图像路径

        Returns:
            预处理后的图像数组 [H, W]，布尔类型 (True为有效区域，False为背景)
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)   # 把图像加载为灰度格式
        img = self._crop_to_target_size(img)  # 裁剪到目标尺寸

        # 直接转换为布尔型，节省内存
        # 背景是0，物体是255，所以只要 >0 就是有效区域
        return torch.from_numpy(img > 0)


    
    def _load_normal(self, path: Union[str, Path]) -> torch.Tensor:
        """
        加载并预处理法向量图
        
        Args:
            path: 法向量图路径
            
        Returns:
            预处理后的法向量张量 [3, H, W]，单位向量，范围 [-1, 1]
        """
        img = read_image_cv2(str(path), rgb=True)   # 把图像加载为RGB格式
        img = self._crop_to_target_size(img)  # 裁剪到目标尺寸

        # 将RGB值转换为[-1, 1]范围的法向量
        max_value = calc_max_value(img.dtype)
        normal = torch.from_numpy(img).permute(2, 0, 1).to(torch.get_default_dtype()).div(max_value).mul(2).sub(1)
        normal = self._normalize_gt(normal)  # 归一化为单位向量
        return normal
    
    def _convert_dtype(self, images: list[np.ndarray]) -> torch.Tensor:
        """
        将numpy数组转换为torch.Tensor，并进行归一化

        Args:
            images: 输入图像数组列表 [N, H, W, C]

        Returns:
            预处理后的图像张量 [N, C, H, W]，范围 [0, 1]
        """
        dtype = images[0].dtype
        max_value = calc_max_value(dtype)

        # 将列表转换为numpy数组 [N, H, W, C]
        images_array = np.stack(images, axis=0)

        # [N, H, W, C] to [N, C, H, W]
        images = torch.from_numpy(images_array).permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(max_value)
        return images
    
    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        """
        获取数据集中的一个样本（子类必须实现）
        
        Args:
            idx: 样本索引
            
        Returns:
            包含数据的字典，至少包含'images', 'normal', 'mask'
        """
        raise NotImplementedError("__getitem__ must be implemented by subclass")
    
    def _crop_to_target_size(self, img: np.ndarray) -> np.ndarray:
        """
        中心裁剪图像到目标尺寸
        
        Args:
            img: 输入图像数组 [H, W, C]
            
        Returns:
            裁剪后的图像数组 [img_size, img_size, C]
        """
        return img
    
    def _normalize_gt(self, gt: torch.Tensor) -> torch.Tensor:
        """
        归一化ground truth数据（如法向量）

        Args:
            gt: 输入的ground truth张量 [C, H, W]

        Returns:
            归一化后的ground truth张量 [C, H, W]
        """
        return gt

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

    img = cv2.imread(path)
    if img is None:
        logger.info(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Retry failed for image={path}.")

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def calc_max_value(dtype: np.dtype) -> float:
    """
    Calculate the maximum possible value for a given numpy data type.

    Args:
        dtype (np.dtype): The data type to calculate the max value for.
    """
    if dtype == np.uint8:
        return 255.0
    elif dtype == np.uint16:
        return 65535.0
    elif dtype == np.int32:
        return 2147483647.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")