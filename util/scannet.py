import os
import numpy as np
import SharedArray as SA
import torch
from torch.utils.data import Dataset
import glob

# 注意：确保这些工具函数能处理返回的额外索引信息
from util.voxelize import voxelize
from util.data_util import sa_create
# 如果 data_prepare_scannet 不支持索引透传，建议检查该函数实现
from util.data_util import data_prepare_scannet

class Scannetv2(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1, labeled_point='0.1%'):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point

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
        data = torch.load(data_path)

        # ScanNet 原始数据通常包含: coord, feat, label
        coord, feat, label = data[0], data[1], data[2]
        
        # 1. 记录原始点云的总数（用于 PCL 缓冲区对齐）
        full_size = coord.shape[0]
        
        # 2. 生成原始索引 (0 到 N-1)
        # 这一步至关重要，当经过 voxelize 下采样后，我们需要知道留下的点原本在哪个位置
        point_idx = np.arange(full_size)

        # 3. 弱监督标注处理 (使用 -100 作为忽略索引)
        if self.split == 'train' or self.split == 'trainval':
            if isinstance(self.labeled_point, str) and '%' in self.labeled_point:
                r = float(self.labeled_point.replace('%', '')) / 100
                num_with_anno = max(int(full_size * r), 1)
            else:
                num_with_anno = int(float(self.labeled_point))
            
            num_without_anno = full_size - num_with_anno
            if num_without_anno > 0:
                idx_without_anno = np.random.choice(full_size, num_without_anno, replace=False)
                label[idx_without_anno] = -100 # 设置为忽略标签

        # 4. 数据准备（体素化、增强等）
        # 核心：必须确保 data_prepare_scannet 能够接收并返回 point_idx
        # 如果你的 data_prepare_scannet 没改过，可能需要修改它以支持 point_idx 的同步采样
        coord, feat, label, point_idx = data_prepare_scannet(
            coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index, point_idx
        )

        # 5. 返回 PCL 训练所需的 7 个参数
        # 返回格式: (坐标, 特征, 标签, 长度, 点云ID, 原始索引, 总点数)
        # 注意：这里返回的 label 长度应与采样后的 coord 一致
        return coord, feat, label, len(coord), data_idx, point_idx, full_size

    def __len__(self):
        return len(self.data_list) * self.loop
