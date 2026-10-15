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
        
        # 核心修复1：保留原始类型，避免字符串转换的歧义，后续单独判断%
        self.labeled_point = labeled_point
        # 预解析带%的比例（提前转成浮点数，避免重复解析）
        if isinstance(self.labeled_point, str) and '%' in self.labeled_point:
            self.ratio = float(self.labeled_point[:-1]) / 100
            self.is_global_ratio = True  # 整个点云的比例
        elif isinstance(self.labeled_point, (int, float)):
            self.is_global_ratio = False  # 按类别处理（比例/固定数量）
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
        
        # [新增 1] 获取该房间原始完整点数 (用于Refinery初始化全局空间)
        total_points_in_room = data.shape[0]

        # ========== labeled point处理：修复浮点数问题 ==========
        if self.split in ['train', 'trainval']:
            if self.is_global_ratio:
                # 逻辑1：带%的全局比例（整个点云的比例）
                num_pts = data.shape[0]
                num_with_anno = max(int(num_pts * self.ratio), 1)  # 转为整数，至少1个点
                num_without_anno = num_pts - num_with_anno
                # 边界检查：避免num_without_anno为负数
                if num_without_anno > 0:
                    idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                    data[idx_without_anno, 6] = 255  # 未标注点设为255
            else:
                # 逻辑2：不带%的数值（区分比例<1/固定数量≥1）
                labeled_val = self.labeled_value
                for i in self.num_classes:
                    ind_per_class = np.where(data[:, 6] == i)[0]  # 该类别的所有点索引
                    num_per_class = len(ind_per_class)
                    if num_per_class == 0:
                        continue  # 跳过无点的类别
                    
                    if labeled_val < 1.0:
                        # 子逻辑1：小数→按该类别的比例标注
                        num_with_anno = max(int(num_per_class * labeled_val), 1)  # 转为整数，至少1个点
                    else:
                        # 子逻辑2：整数→按固定数量标注
                        num_with_anno = min(int(labeled_val), num_per_class)  # 不超过类别总点数
                    
                    num_without_anno = num_per_class - num_with_anno
                    # 边界检查：仅当需要隐藏点时才采样
                    if num_without_anno > 0:
                        idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                        data[idx_without_anno, 6] = 255
                    elif num_without_anno == 0:
                        continue  # 所有点都标注，无需处理
                    else:
                        # 标注数量超过类别总点数→所有点都标注
                        pass
        # ====================================================

        # 分离坐标、特征、标签
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        
        # [修改 2] 调用 data_prepare 接收 4 个返回值 (包含 point_idxs)
        # 注意：你需要确保 util/data_util.py 中的 data_prepare 已经按照之前的说明修改过
        coord, feat, label, point_idxs = data_prepare(
            coord, feat, label, self.split, self.voxel_size, 
            self.voxel_max, self.transform, self.shuffle_index
        )
        
        # ========== 生成cloud_id（用于伪标签追踪）==========
        # 方案1：使用文件路径哈希（推荐，稳定且唯一）
        file_name = self.data_list[data_idx]
        cloud_id = hash(file_name) & 0x7FFFFFFF  # 转为非负int32
        
        # ========== 修改返回值（7个元素）==========
        # 1. coord: 裁剪后的坐标
        # 2. feat: 裁剪后的特征
        # 3. label: 裁剪后的标签
        # 4. length: 裁剪后的点数 (offset用)
        # 5. cloud_id: 房间唯一ID
        # 6. point_idxs: 裁剪后的点在原始大图中的索引 [新增]
        # 7. total_points_in_room: 原始大图的总点数 [新增]
        return coord, feat, label, coord.shape[0], cloud_id, point_idxs, total_points_in_room

    def __len__(self):
        return len(self.data_idx) * self.loop
