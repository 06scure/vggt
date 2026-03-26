#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import logging
import glob
from pathlib import Path

from training.data.datasets.ps_dataset import PSDataset


class DiLiGenTDataset(PSDataset):
    """
    DiLiGenT 数据集类

    数据集路径: /home/user/dataset/DiLiGenT_518
    图像分辨率: 518*518
    每个item有96张图像，共10个item
    测试数据集
    """
    def __init__(
        self,
        common_conf,
        split: str = "test",
        DILIGENT_DIR: str = "/home/user/dataset/DiLiGenT_518",
        img_per_seq: int = 96,
        len_test: int = 10,
    ):
        """
        初始化DiLiGenT数据集

        Args:
            common_conf: 通用配置对象
            split: 数据集分割类型 ("train" 或 "test")
            DILIGENT_DIR: 数据集根目录路径
            img_per_seq: 每个序列抽取的图像数量
            len_test: 测试集长度
        """
        super().__init__(
            common_conf=common_conf,
            split=split,
            img_per_seq=img_per_seq,
            len_train=len_test,
            len_test=len_test,
        )

        self.DILIGENT_DIR = Path(DILIGENT_DIR)
        self.pmsData_dir = self.DILIGENT_DIR / "pmsData"

        if not self.DILIGENT_DIR.exists() or not self.pmsData_dir.exists():
            raise FileNotFoundError(f"DiLiGenT dataset not found at {DILIGENT_DIR}")

        # 查找所有序列
        self._find_sequences()

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: DiLiGenT Data size: {len(self.sequence_list)}")
        logging.info(f"{status}: DiLiGenT Data dataset length: {len(self)}")

    def _find_sequences(self):
        """查找所有序列目录"""
        # 查找所有物体目录作为序列
        seq_dirs = [d for d in self.pmsData_dir.iterdir() if d.is_dir() and d.name.endswith('PNG')]
        seq_dirs.sort()

        for seq_dir in seq_dirs:
            seq_name = seq_dir.name
            seq_data = self._load_sequence(seq_dir, seq_name)
            if seq_data is not None:
                self.data_store[seq_name] = seq_data
                self.sequence_list.append(seq_name)

    def _load_sequence(self, seq_dir, seq_name):
        """
        加载单个序列的数据

        Args:
            seq_dir: 序列目录路径
            seq_name: 序列名称

        Returns:
            序列数据字典，如果无效则返回None
        """
        # 查找所有图像文件 (png格式，001.png到096.png)
        images = []
        for i in range(1, 97):
            img_path = seq_dir / f"{i:03d}.png"
            if img_path.exists():
                images.append(str(img_path))

        if len(images) == 0:
            logging.warning(f"No images found in {seq_dir}")
            return None

        # 查找法向量真值
        gt_normal = None
        normal_names = ["Normal_gt.png", "normal_gt.png", "gt_normal.png"]
        for name in normal_names:
            normal_path = seq_dir / name
            if normal_path.exists():
                gt_normal = str(normal_path)
                break

        # 查找mask
        mask = None
        mask_names = ["mask.png", "object_mask.png"]
        for name in mask_names:
            mask_path = seq_dir / name
            if mask_path.exists():
                mask = str(mask_path)
                break

        seq_data = {
            "images": images,
            "gt_normal": gt_normal,
            "mask": mask,
        }

        return seq_data