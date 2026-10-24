import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SemanticKITTIDataset(Dataset):
    def __init__(self, split='train', data_root=None, test_area=None, voxel_size=0.05, voxel_max=80000, transform=None, shuffle_index=False, loop=1, labeled_point='0.1%'):
        self.data_root = data_root
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.loop = loop
        self.labeled_point = labeled_point
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        else:
            self.seqs = ['08'] 
        self.data_list = []
        for seq in self.seqs:
            seq_path = os.path.join(data_root, 'sequences', seq)
            if not os.path.exists(seq_path):
                seq_path_alt = os.path.join(data_root, seq)
                if os.path.exists(seq_path_alt):
                    seq_path = seq_path_alt
                else:
                    raise FileNotFoundError(f"找不到序列文件夹: {seq_path}")

            velodyne_path = os.path.join(seq_path, 'velodyne')
            
            if not os.path.exists(velodyne_path):
                 raise FileNotFoundError(f"找不到 velodyne 文件夹: {velodyne_path}")

            files = sorted(os.listdir(velodyne_path))
            for f in files:
                self.data_list.append({
                    'pc_path': os.path.join(velodyne_path, f),
                    'lb_path': os.path.join(seq_path, 'labels', f.replace('.bin', '.label')),
                    'clip_path': os.path.join(seq_path, 'clip_pseudo', f.replace('.bin', '.npy'))
                })
        self.learning_map = {
            0 : 255,     # "unlabeled"
            1 : 255,     # "outlier"
            10: 0,       # "car"
            11: 1,       # "bicycle"
            13: 5,       # "bus" -> mapped to "other-vehicle" usually
            15: 2,       # "motorcycle"
            18: 3,       # "truck"
            20: 4,       # "other-vehicle"
            30: 5,       # "person"
            31: 6,       # "bicyclist"
            32: 7,       # "motorcyclist"
            40: 8,       # "road"
            44: 9,       # "parking"
            48: 10,      # "sidewalk"
            49: 11,      # "other-ground"
            50: 12,      # "building"
            51: 13,      # "fence"
            52: 255,     # "other-structure"
            60: 8,       # "lane-marking" -> to "road" 
            70: 14,      # "vegetation"
            71: 15,      # "trunk"
            72: 16,      # "terrain"
            80: 17,      # "pole"
            81: 18,      # "traffic-sign"  <-- 必须是 18，不能是 19
            99: 255,     # "other-object"
            252: 255, 253: 255, 254: 255, 255: 255, 256: 255, 257: 255, 258: 255, 259: 255
        }

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, index):
        item = self.data_list[index % len(self.data_list)]
        
        data = np.fromfile(item['pc_path'], dtype=np.float32).reshape(-1, 4)
        coord = data[:, :3]
        feat = data[:, 3].reshape(-1, 1) 
        
        if os.path.exists(item['lb_path']):
            label_raw = np.fromfile(item['lb_path'], dtype=np.uint32).reshape(-1)
            label_sem = label_raw & 0xFFFF
            label = np.vectorize(self.learning_map.get)(label_sem, 255)
        else:
            label = np.zeros(coord.shape[0], dtype=np.int32) + 255
        
        if os.path.exists(item['clip_path']):
            try:
                clip_label = np.load(item['clip_path'])
            except:
                clip_label = np.full(label.shape, 255, dtype=np.uint8)
        else:
            clip_label = np.full(label.shape, 255, dtype=np.uint8)
        if self.split == 'train':
            rng = np.random.default_rng(index)
            mask = rng.random(label.shape) < 0.001 
            label[~mask] = 255
        coord, feat, label, clip_label = self.voxelize(coord, feat, label, clip_label)
        return coord, feat, label, clip_label
    def voxelize(self, coord, feat, label, clip_label):
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx = (coord / self.voxel_size).astype(np.int32)
        
        _, uniq_idx = np.unique(idx, axis=0, return_index=True)
        
        coord, feat, label, clip_label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx], clip_label[uniq_idx]
        
        if coord.shape[0] > self.voxel_max:
            shuf = np.random.choice(coord.shape[0], self.voxel_max, replace=False)
            coord, feat, label, clip_label = coord[shuf], feat[shuf], label[shuf], clip_label[shuf]
            
        return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), torch.LongTensor(clip_label)
