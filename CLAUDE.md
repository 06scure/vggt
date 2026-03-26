# 项目说明

## 开发约定
1. 在代码中，要遵循软件开发的“高内聚低耦合”的原则，首先搜索项目中是否有类似的功能已经实现，如有实现，请继承后在实现特定功能；
2. 在代码中，要适当添加中文注释。增加了新的函数/类/文件后，要增加使用示例；
3. 如无必要，不要增加文件。如不得不增加临时测试脚本，请放到 test/ 文件夹下；
4. 充分理解用户的意图和需求，根据当前代码和情况去修改代码，不做过多的修改。如果认为有地方需要修改或者优化，应先询问用户的意愿，而不是直接修改；
5. 如果不确定文件内容或代码结构，请使用工具查询相关信息，绝对不要猜测或编造答案，可以对用户发起提问；
6. 在写代码的时候，会对整体内容做一个完整的规划，避免出现前面写了一个方法，但是后面完全没调用的情况；
7. 在修改用户的代码的时候，如果修改内容较少，一般会直接反馈给用户。但如果修改内容很多，则会直接返回整个类/函数/文件；
8. 如需测试，使用conda py310环境。
---
## 项目目标

**核心任务**: 将 VGGT (Visual Geometry Grounded Transformer) 从多视角立体 (MVS) 任务迁移到光度立体 (Photometric Stereo) 任务。冻结原模型权重，仅使用输出的第0帧token，训练一个新的dpt head用于预测法向量。

**输入**: N 张同一物体在不同光照条件下的图像（固定视角）
**输出**: 单张表面法向量图 (Surface Normal Map)

---

## 核心概念映射

我们进行的是"任务迁移"，即输出的物理意义改变，但架构保持基本不变。

| 组件 | 原VGGT (MVS任务) | PS-VGGT (PS任务) | 处理方式 |
|:---|:---|:---|:---|
| **输入语义** | N 个不同 **视角** | N 个不同 **光照** | **保持形状**。输入张量 `[B, N, 3, H, W]` 仍然有效 |
| **输入Token** | 图像Patch + **相机Token** | 图像Patch | **忽略**。现阶段忽略相机Token |
| **核心结构** | 交替注意力 (帧内 + 全局) | 相同 | **保留**。这能有效聚合物体几何特征 |
| **输出头** | 相机姿态 + 深度/点图 | **法向量图** | **新增**。添加法向量头 |
| **监督信号** | 姿态损失 + 深度损失 | **法向量损失** (MSE) | **新增**。改变损失函数 |

---

##  实现进度总览

#### 1. 模型架构 
- **[vggt/models/vggt.py](vggt/models/vggt.py)**
  - VGGT 主模型类
  - 保留 Aggregator (DINOv2-Large ViT + 交替注意力Transformer)
  - 添加: NormalHead (基于DPT的法向量预测头)
  - 支持预训练权重加载
  - 支持 freeze aggregator

- **[vggt/heads/normal_head.py](vggt/heads/normal_head.py)**
  - 基于DPT架构的法向量预测头
  - 输出: 3通道法向量
  - 归一化到单位长度
  - 使用第0帧的特征frame_attention进行预测

#### 2. 数据集 
- **[training/data/datasets/ps_dataset.py](training/data/datasets/ps_dataset.py)**
  - 基类数据集
  - 本方法为非校准的光度立体法，仅读取一组图像、法向量真值(gt_normal)、mask蒙版数据。
  - 在文件夹中从随机抽取图像(防止数据太多而OOM)

- **[training/data/datasets/ps_wild.py](training/data/datasets/ps_wild.py)**
  - 路径在 /home/user/dataset/PSWild
  - 图像分辨率为512*512
  - 继承自基类数据集
  - 每个item有10张图像，约10000个item
  - 禁用数据增强（如裁剪、缩放）

- **[training/data/datasets/ps_diligent.py](training/data/datasets/ps_diligent.py)**
  - 路径在 /home/user/dataset/DiLiGenT_518
  - 图像分辨率为518*518
  - DiLiGenTDataset 数据加载器
  - 继承自基类数据集
  - 每个item有96张图像，共10个item
  - 测试数据集

#### 3. 损失函数
- **[training/ps_loss.py](training/ps_loss.py)**
  - MSE计算损失训练
  - Mask外的背景信息不计算损失

#### 4. 训练脚本
- **[training/train.py](training/train.py)**
  - 训练脚本

- **[training/eval.py](training/eval.py)**
  - 评估脚本

---

## 数据流图

```
输入: [B, N, 3, H, W]
         ↓ (N个不同光照的图像)
    ┌─────────────────────────────┐
    │   Aggregator (DINOv2 ViT)   │
    │  - 帧内注意力 (每帧独立)      │
    │  - 全局注意力 (跨帧聚合)      │
    └─────────────────────────────┘
         ↓
    aggregated_tokens: [B, N, L, C]
         ↓
    ┌─────────────────────────────┐
    │    NormalHead (DPT)         │
    │  - 多尺度特征融合            │
    │  - 密集预测解码              │
    └─────────────────────────────┘
         ↓
    法向量: [B, 3, H, W] (单位向量)
```

---

## 关键设计决策

### 1. 为什么保留Aggregator?

Aggregator 中的交替注意力机制:
- **帧内注意力**: 提取每个光照图像的局部特征
- **全局注意力**: 跨图像聚合信息，包括物体几何特征、帧顺序等

### 2. 为什么冻结Backbone?

- 数据集规模有限
- 从头训练容易过拟合
- 预训练的模型已理解几何、光源等的对应关系
- 只需训练NormalHead学习特征→法向量的映射

---
## 数据集目录

 - /home/user/dataset/DiLiGenT
 - /home/user/dataset/PSWild

 ## 权重文件目录
 - /home/user/dataset/ckpt/model.pt
---

## 注意事项

 - 显卡为5070ti(16G),要注意可能会OOM(out of memory)，在保证模型效果的同时尽量节约内存
 - 模型需要训练的参数较少，训练脚本应小批次训练快速验证
 - ViT中已经对RGB自动归一化，注意图像读取维度，RGB的通道顺序问题