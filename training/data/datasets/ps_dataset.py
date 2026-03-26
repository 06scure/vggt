#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import logging
import random
import numpy as np
import cv2

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import read_image_cv2


class PSDataset(BaseDataset):
    """
    光度立体(Photometric Stereo)数据集基类

    本方法为非校准的光度立体法，仅读取一组图像、法向量真值(gt_normal)、mask蒙版数据。
    在文件夹中从随机抽取图像(防止数据太多而OOM)
    """
    def __init__(
        self,
        common_conf,
        split: str = "train",
        img_per_seq: int = 10,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        初始化PS数据集基类

        Args:
            common_conf: 通用配置对象
            split: 数据集分割类型 ("train" 或 "test")
            img_per_seq: 每个序列抽取的图像数量
            len_train: 训练集长度
            len_test: 测试集长度
        """
        super().__init__(common_conf=common_conf)

        self.split = split
        self.img_per_seq = img_per_seq
        self.training = split == "train"

        if self.training:
            self.len_train = len_train
        else:
            self.len_train = len_test

        self.data_store = {}
        self.sequence_list = []

        logging.info(f"PSDataset initialized for split: {split}")

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx):
        """
        获取数据集的一个item

        Args:
            idx: 序列索引 (整数)

        Returns:
            包含图像、法向量真值、mask等数据的字典
        """
        return self.get_data(seq_index=idx, img_per_seq=self.img_per_seq, aspect_ratio=1.0)

    def get_data(self, seq_index: int = None, img_per_seq: int = None, seq_name: str = None, ids: list = None, aspect_ratio: float = 1.0):
        """
        读取一个序列的数据

        Args:
            seq_index: 序列索引
            img_per_seq: 每个序列抽取的图像数量
            seq_name: 序列名称
            ids: 特定的图像id列表
            aspect_ratio: 宽高比

        Returns:
            包含图像、法向量真值、mask等数据的字典
        """
        if img_per_seq is None:
            img_per_seq = self.img_per_seq

        if seq_name is None:
            seq_index = seq_index % len(self.sequence_list)
            seq_name = self.sequence_list[seq_index]

        seq_data = self.data_store[seq_name]

        # 随机抽取图像
        if ids is None:
            num_images = len(seq_data["images"])
            ids = np.random.choice(num_images, img_per_seq, replace=False)

        # 读取图像
        images = []
        image_paths = []
        for idx in ids:
            img_path = seq_data["images"][idx]
            image = read_image_cv2(img_path)
            if image is not None:
                images.append(image)
                image_paths.append(img_path)

        # 读取法向量真值和mask
        gt_normal = None
        if "gt_normal" in seq_data and seq_data["gt_normal"] is not None:
            gt_normal_path = seq_data["gt_normal"]
            gt_normal = cv2.imread(gt_normal_path, cv2.IMREAD_UNCHANGED)
            if len(gt_normal.shape) == 3 and gt_normal.shape[2] == 4:
                gt_normal = gt_normal[:, :, :3]  # 去除alpha通道

            # 根据数据类型解码法向量
            if gt_normal.dtype == np.uint8:
                # DiLiGenT数据集: uint8, [0, 255] -> [-1, 1]
                gt_normal = (gt_normal.astype(np.float32) - 128.0) / 128.0
            elif gt_normal.dtype == np.uint16:
                # PSWild数据集: uint16, [0, 65535] -> [-1, 1]
                gt_normal = (gt_normal.astype(np.float32) - 32768.0) / 32768.0
            # 确保法向量是单位长度
            if len(gt_normal.shape) == 3:
                norms = np.linalg.norm(gt_normal, axis=2, keepdims=True)
                gt_normal = gt_normal / (norms + 1e-8)

        mask = None
        if "mask" in seq_data and seq_data["mask"] is not None:
            mask_path = seq_data["mask"]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128

        # 计算目标形状（确保是 patch 大小的倍数）
        patch_size = self.patch_size
        processed_images = []
        original_sizes = []

        if images:
            h, w, c = images[0].shape
            # 计算小于等于原尺寸的最大14的倍数
            target_h = (h // patch_size) * patch_size
            target_w = (w // patch_size) * patch_size
            # 中心裁剪
            start_h = max(0, (h - target_h) // 2)
            start_w = max(0, (w - target_w) // 2)
            end_h = start_h + target_h
            end_w = start_w + target_w

            # 裁剪所有图像
            for img in images:
                original_size = np.array(img.shape[:2])
                original_sizes.append(original_size)
                processed_img = img[start_h:end_h, start_w:end_w, :]
                processed_images.append(processed_img)

            # 裁剪 gt_normal
            if gt_normal is not None:
                gt_normal = gt_normal[start_h:end_h, start_w:end_w, :]

            # 裁剪 mask
            if mask is not None:
                mask = mask[start_h:end_h, start_w:end_w]

        batch = {
            "seq_name": seq_name,
            "ids": ids,
            "frame_num": len(processed_images),
            "images": processed_images,
            "gt_normal": gt_normal,
            "mask": mask,
            "original_sizes": original_sizes,
            "image_paths": image_paths,
        }

        return batch

    def _load_sequence(self, seq_dir, seq_name):
        """
        加载单个序列的数据

        Args:
            seq_dir: 序列目录
            seq_name: 序列名称

        Returns:
            序列数据字典
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _find_sequences(self, root_dir):
        """
        查找所有序列

        Args:
            root_dir: 根目录

        Returns:
            序列列表
        """
        raise NotImplementedError("Subclasses must implement this method")