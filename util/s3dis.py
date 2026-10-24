import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, 
                 voxel_max=None, transform=None, shuffle_index=False, loop=1, labeled_point=0.1):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.transform = transform
        self.voxel_max = voxel_max
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point
        if isinstance(self.labeled_point, str) and '%' in self.labeled_point:
            self.ratio = float(self.labeled_point[:-1]) / 100
            self.is_global_ratio = True  
        elif isinstance(self.labeled_point, (int, float)):
            self.is_global_ratio = False  
            self.labeled_value = float(self.labeled_point)
        else:
            raise TypeError(f"labeled_point仅支持int/float/带%的str，当前类型：{type(self.labeled_point)}")
        
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]

        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7 (x,y,z,r,g,b,label)
                sa_create("shm://{}".format(item), data)
        
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

        self.num_classes = list(range(13)) + [255]

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        total_points_in_room = data.shape[0]
        if self.split in ['train', 'trainval']:
            if self.is_global_ratio:
                num_pts = data.shape[0]
                num_with_anno = max(int(num_pts * self.ratio), 1)  
                num_without_anno = num_pts - num_with_anno
                if num_without_anno > 0:
                    idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                    data[idx_without_anno, 6] = 255  
            else:
                labeled_val = self.labeled_value
                for i in self.num_classes:
                    ind_per_class = np.where(data[:, 6] == i)[0]  
                    num_per_class = len(ind_per_class)
                    if num_per_class == 0:
                        continue  
                    
                    if labeled_val < 1.0:
                        num_with_anno = max(int(num_per_class * labeled_val), 1)  
                    else:
                        num_with_anno = min(int(labeled_val), num_per_class)  
                    
                    num_without_anno = num_per_class - num_with_anno
                    if num_without_anno > 0:
                        idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                        data[idx_without_anno, 6] = 255
                    elif num_without_anno == 0:
                        continue  
                    else:
                        pass
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label, point_idxs = data_prepare(
            coord, feat, label, self.split, self.voxel_size, 
            self.voxel_max, self.transform, self.shuffle_index
        )
        file_name = self.data_list[data_idx]
        cloud_id = hash(file_name) & 0x7FFFFFFF 
        return coord, feat, label, coord.shape[0], cloud_id, point_idxs, total_points_in_room
    def __len__(self):
        return len(self.data_idx) * self.loop
