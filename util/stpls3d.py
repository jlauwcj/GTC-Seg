import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
from util.data_util import sa_create
from util.data_util import data_prepare_stpls3d
from util.ply import write_ply, read_ply
import os

train_sequences = ['RealWorldData', 'Synthetic_v1', 'Synthetic_v2', 'Synthetic_v3']
cvalid_sequences = ['OCCC_points', 'RA_points', 'USC_points', 'WMSC_points']

def changeSemLabels(cloud):
    """将原始标签映射为6类语义分割任务的标签"""
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 2) & (cloud[:, 6:7] <= 4), 2, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 5) & (cloud[:, 6:7] <= 6), 3, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 8), 3, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 11) & (cloud[:, 6:7] <= 12), 4, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 14), 5, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 7) & (cloud[:, 6:7] <= 10), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 13), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 15) & (cloud[:, 6:7] <= 16), 0, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 17), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] > 17), 0, cloud[:, 6:7])
    return cloud

def ply2array(ply_path):
    """读取Ply文件并转换为numpy数组，同时转换标签"""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"文件不存在：{ply_path}")
    cloud = read_ply(ply_path)
    cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'], 
                       cloud['red'], cloud['green'], cloud['blue'], 
                       cloud['class'])).T
    cloud = changeSemLabels(cloud)
    return cloud

class STPLS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=0, 
                 voxel_size=0.04, voxel_max=None, transform=None, 
                 shuffle_index=False, loop=1, labeled_point='0.1%', use_shm=False):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.transform = transform
        self.voxel_max = voxel_max
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point
        self.use_shm = use_shm
        
        data_list = []
        for sq in train_sequences:  # 遍历所有数据集目录
            seq_dir = os.path.join(data_root, sq)
            if not os.path.exists(seq_dir):
                print(f"警告: 目录 {seq_dir} 不存在，跳过")
                continue
            seq_files = [f for f in os.listdir(seq_dir) if f.endswith('.ply')]
            for f in seq_files:
                data_list.append((f[:-4], sq))  # 保存（文件名无后缀，所属目录）
        
        # 根据split参数划分训练集和验证集
        target_valid_seq = cvalid_sequences[test_area]  # 如'OCCC_points'
        if split == 'train':
            # 训练集：排除验证集序列中的样本
            self.data_list = [(f, sq) for f, sq in data_list if target_valid_seq not in f]
        else:
            # 验证集：只包含验证集序列中的样本
            self.data_list = [(f, sq) for f, sq in data_list if target_valid_seq in f]
        
        # 移除文件扩展名并排序（已在元组中处理）
        self.data_list = sorted(self.data_list, key=lambda x: x[0])
        
        # 打印加载的样本数量
        print(f"Totally {len(self.data_list)} samples in {split} set (test_area={test_area})")
        
        # 存储数据集根路径
        self.data_root = data_root
        self.data_idx = np.arange(len(self.data_list))
        if self.use_shm:
            print(f"[{split}] 正在将数据预加载到共享内存 (/dev/shm) 中，请耐心等待...")
            for file_name, directory in self.data_list:
                shm_name = f"shm://stpls3d_{split}_{file_name}"
                try:
                    # 检查是否已存在，防止重复加载报错
                    SA.attach(shm_name)
                except FileNotFoundError:
                    # 不存在则读取文件并写入共享内存
                    full_path = os.path.join(self.data_root, directory, file_name + '.ply')
                    data = ply2array(full_path)
                    sa_create(shm_name, data)
            print(f"[{split}] 共享内存加载完毕！极速模式已开启！")
    
    def __getitem__(self, idx):
        """获取单个样本"""
        data_idx = self.data_idx[idx % len(self.data_idx)]
        file_name, directory = self.data_list[data_idx]  # 获取文件名和所属目录
        if self.use_shm:
            shm_name = f"shm://stpls3d_{self.split}_{file_name}"
            data = SA.attach(shm_name).copy()
        else:
            full_path = os.path.join(self.data_root, directory, file_name + '.ply')  # 直接用目录构建路径
            
            # 验证路径有效性
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"路径错误或文件不存在：{full_path}")
            
            # 读取并处理数据
            data = ply2array(full_path)
        full_size = data.shape[0]
        point_idx = np.arange(full_size)
        if self.split == 'train' or self.split == 'trainval':
            if isinstance(self.labeled_point, str) and '%' in self.labeled_point:
                r = float(self.labeled_point.replace('%', '')) / 100
                num_with_anno = max(int(full_size * r), 1)
            else:
                num_with_anno = int(float(self.labeled_point))
                
            num_without_anno = full_size - num_with_anno
            if num_without_anno > 0:
                rng = np.random.RandomState(data_idx)
                idx_without_anno = rng.choice(full_size, num_without_anno, replace=False)
                data[idx_without_anno, 6] = -100 
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label, point_idx = data_prepare_stpls3d(
            coord, feat, label, self.split, self.voxel_size, 
            self.voxel_max, self.transform, self.shuffle_index, point_idx
        )
        return coord, feat, label, len(coord), data_idx, point_idx, full_size
    
    def __len__(self):
        return len(self.data_idx) * self.loop
