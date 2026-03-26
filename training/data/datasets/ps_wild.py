#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import logging
import glob
from pathlib import Path

from training.data.datasets.ps_dataset import PSDataset


class PSWildDataset(PSDataset):
    """
    PSWild数据集类

    数据集路径: /home/user/dataset/PSWild
    图像分辨率: 512*512
    每个item有10张图像，约10000个item
    禁用数据增强（如随机裁剪、缩放）

    数据结构:
        PSWild/
            american_football_ball.obj_xxxxx.data/
                00000.tif
                00001.tif
                ...
                00009.tif
                normal.tif
                mask.png
    """
    def __init__(
        self,
        common_conf,
        split: str = "train",
        PSWILD_DIR: str = "/home/user/dataset/PSWild",
        img_per_seq: int = 10,
        len_train: int = 10000,
        len_test: int = 1000,
    ):
        """
        初始化PSWild数据集

        Args:
            common_conf: 通用配置对象
            split: 数据集分割类型 ("train" 或 "test")
            PSWILD_DIR: 数据集根目录路径
            img_per_seq: 每个序列抽取的图像数量
            len_train: 训练集长度
            len_test: 测试集长度
        """
        super().__init__(
            common_conf=common_conf,
            split=split,
            img_per_seq=img_per_seq,
            len_train=len_train,
            len_test=len_test,
        )

        self.PSWILD_DIR = Path(PSWILD_DIR)

        if not self.PSWILD_DIR.exists():
            raise FileNotFoundError(f"PSWild dataset not found at {PSWILD_DIR}")

        # 查找所有序列
        self._find_sequences()

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: PSWild Data size: {len(self.sequence_list)}")
        logging.info(f"{status}: PSWild Data dataset length: {len(self)}")

    def _find_sequences(self):
        """查找所有序列目录"""
        root_dir = self.PSWILD_DIR

        # 查找所有子目录作为序列
        seq_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.endswith('.data')]
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
        # 查找所有图像文件 (tif格式)
        images = []
        for i in range(10):
            img_path = seq_dir / f"{i:05d}.tif"
            if img_path.exists():
                images.append(str(img_path))

        if len(images) == 0:
            logging.warning(f"No images found in {seq_dir}")
            return None

        # 查找法向量真值
        gt_normal = None
        normal_path = seq_dir / "normal.tif"
        if normal_path.exists():
            gt_normal = str(normal_path)

        # 查找mask
        mask = None
        mask_path = seq_dir / "mask.png"
        if mask_path.exists():
            mask = str(mask_path)

        seq_data = {
            "images": images,
            "gt_normal": gt_normal,
            "mask": mask,
        }

        return seq_data