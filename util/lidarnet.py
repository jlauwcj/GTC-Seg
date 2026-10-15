import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

# 注意：确保这些工具函数能处理返回的额外索引信息
from util.voxelize import voxelize
from util.data_util import sa_create
from util.data_util import data_prepare_scannet

class Lidarnet(Dataset):
    def __init__(self, split='train', data_root='dataset/lidarnet', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1, labeled_point='0.1%', ignore_label=255):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point
        # 动态接收 YAML 里配置的 ignore_label (默认是 24 或 255)
        self.ignore_label = ignore_label

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))
            
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        
        # 1. 读取转换好的 4 元组数据 (coord, color, sem, ins)
        data = torch.load(data_path)
        coord, feat, label = data[0], data[1], data[2]
        
        # LiDAR-Net 的 color 范围是 0-255，将其归一化到 -1 到 1 (与网络原始分布对齐)
        feat = (feat / 127.5) - 1.0
        
        # 记录原始点云的总数（用于 PCL 缓冲区对齐）
        full_size = coord.shape[0]
        # 生成原始索引 (0 到 N-1)
        point_idx = np.arange(full_size)

        # 终极防爆盾：把超出 0~23 范围的非法标签(包括原本的 24)，统一设为官方配置的 ignore_label
        label[(label < 0) | (label >= 24)] = self.ignore_label

        # 3. 弱监督标注处理 (随机丢弃标签)
        if self.split == 'train' or self.split == 'trainval':
            # 找出真正有有效标签的点（排除掉原本就是 unknown 的点）
            valid_mask = (label != self.ignore_label)
            valid_indices = np.where(valid_mask)[0]
            valid_size = len(valid_indices)
            
            if valid_size > 0:
                if isinstance(self.labeled_point, str) and '%' in self.labeled_point:
                    r = float(self.labeled_point.replace('%', '')) / 100
                    num_with_anno = max(int(valid_size * r), 1)
                else:
                    num_with_anno = int(float(self.labeled_point))
                
                num_without_anno = valid_size - num_with_anno
                if num_without_anno > 0:
                    # 仅在有效标签中随机抽取点变为无标注
                    idx_without_anno = np.random.choice(valid_indices, num_without_anno, replace=False)
                    label[idx_without_anno] = self.ignore_label

        # 4. 数据准备（体素化、增强等）
        coord, feat, label, point_idx = data_prepare_scannet(
            coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index, point_idx
        )

        # 5. 返回 PCL 训练所需的 7 个参数
        return coord, feat, label, len(coord), data_idx, point_idx, full_size

    def __len__(self):
        return len(self.data_list) * self.loop
