import os
import numpy as np
import SharedArray as SA
import torch
from torch.utils.data import Dataset
import glob
from util.voxelize import voxelize
from util.data_util import sa_create
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
        coord, feat, label = data[0], data[1], data[2]
        full_size = coord.shape[0]
        point_idx = np.arange(full_size)
        if self.split == 'train' or self.split == 'trainval':
            if isinstance(self.labeled_point, str) and '%' in self.labeled_point:
                r = float(self.labeled_point.replace('%', '')) / 100
                num_with_anno = max(int(full_size * r), 1)
            else:
                num_with_anno = int(float(self.labeled_point))
            
            num_without_anno = full_size - num_with_anno
            if num_without_anno > 0:
                idx_without_anno = np.random.choice(full_size, num_without_anno, replace=False)
                label[idx_without_anno] = -100 
        coord, feat, label, point_idx = data_prepare_scannet(
            coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index, point_idx
        )
        return coord, feat, label, len(coord), data_idx, point_idx, full_size

    def __len__(self):
        return len(self.data_list) * self.loop
