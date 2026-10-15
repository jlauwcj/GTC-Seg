import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, target = list(zip(*batch))
    offset, count = [], 0
    
    for item in coord:
        #batch_size+=1
        count += item.shape[0]
        offset.append(count)
    
    return torch.cat(coord), torch.cat(feat), torch.cat(target), torch.IntTensor(offset)
    
def collate_fn_shapenet(batch):
    coord, feat, target, label = list(zip(*batch))
    offset, count, batch_size = [], 0, 0
    
    batch_size=len(coord)
    lstCoordSize=[]
    for item in coord:
        #batch_size+=1
        count += item.shape[0]
        offset.append(count)
        lstCoordSize.append(item.shape[0])
    
    return batch_size, lstCoordSize, torch.cat(coord), torch.cat(feat), torch.cat(target), torch.IntTensor(offset), torch.cat(label)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# ===================== 新增：S3DIS数据集核心依赖函数（修复导入错误） =====================
def collect_point_label(anno_path, out_filename, class2label):
    """
    从S3DIS的txt标注文件中收集点云和标签，保存为numpy格式
    :param anno_path: S3DIS的txt标注文件路径
    :param out_filename: 输出的npz文件路径
    :param class2label: 类别到标签的映射字典
    :return: 点云数组(N,6)，标签数组(N,)
    """
    points = []
    labels = []
    with open(anno_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) < 7:  # 过滤无效行（需包含x,y,z,r,g,b,label）
                continue
            # S3DIS数据格式：x y z r g b label
            x, y, z = float(line[0]), float(line[1]), float(line[2])
            r, g, b = int(line[3]), int(line[4]), int(line[5])
            label_name = line[6]
            # 未定义的标签归为杂波（clutter）
            if label_name not in class2label:
                label_name = 'clutter'
            points.append([x, y, z, r, g, b])
            labels.append(class2label[label_name])
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # 保存为npz文件（可选）
    np.savez(out_filename, points=points, labels=labels)
    return points, labels

def load_indoor3d_data(anno_path, class2label):
    """
    直接加载S3DIS的txt标注文件，返回点云和标签（s3dis.py的核心依赖）
    :param anno_path: S3DIS的txt标注文件路径
    :param class2label: 类别到标签的映射字典
    :return: 点云数组(N,6)，标签数组(N,)
    """
    points = []
    labels = []
    with open(anno_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) < 7:  # 过滤无效行
                continue
            # 解析坐标和颜色
            x = float(line[0])
            y = float(line[1])
            z = float(line[2])
            r = int(line[3])
            g = int(line[4])
            b = int(line[5])
            # 解析标签（未定义标签归为clutter）
            label_name = line[6]
            if label_name not in class2label:
                label_name = 'clutter'
            label = class2label[label_name]
            points.append([x, y, z, r, g, b])
            labels.append(label)
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return points, labels
# ===================== 新增结束 =====================

def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    # [新增] 生成原始索引 (0, 1, 2, ... N-1)
    idx = np.arange(coord.shape[0])

    if transform:
        coord, feat, label = transform(coord, feat, label)
    
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        # [修改] 同步筛选 idx
        coord, feat, label, idx = coord[uniq_idx], feat[uniq_idx], label[uniq_idx], idx[uniq_idx]
        
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        # [修改] 同步裁剪 idx
        coord, feat, label, idx = coord[crop_idx], feat[crop_idx], label[crop_idx], idx[crop_idx]
        
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        # [修改] 同步打乱 idx
        coord, feat, label, idx = coord[shuf_idx], feat[shuf_idx], label[shuf_idx], idx[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    # [修改] 将 idx 转为 LongTensor 并返回
    idx = torch.LongTensor(idx)
    
    return coord, feat, label, idx

def data_prepare_scannet(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    #label[label == -100] = 255
    #feat = (feat + 1.) * 127.5
    

    if transform:
        coord, feat, label = transform(coord, feat, label)
        #coord, feat = transform(coord, feat)
    
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    
    #feat = feat / 127.5 - 1.
    
    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def data_prepare_stpls3d(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
        #coord, feat = transform(coord, feat)
    
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        #print(max(crop_idx))
        #print('----')
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def data_prepare_sensat(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
        #coord, feat = transform(coord, feat)
    
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        #print(max(crop_idx))
        #print('----')
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    

    coord_min = np.min(coord, 0)
    #print(coord_min)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def data_prepare_shapenet(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, choice=None):
    
    #coord=pc_normalize(coord)

    if transform:
        coord, feat, label = transform(coord, feat, label)
        #coord, feat = transform(coord, feat)
    else:
        #coord = coord.numpy()
        pass
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    #coord_min = np.min(coord, 0)
    #coord -= coord_min
    coord=coord[choice,:]
    feat = feat[choice,:]
    label=label[choice]
    
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label
