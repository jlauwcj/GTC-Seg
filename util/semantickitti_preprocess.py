import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

class SemanticKITTIPreprocessor:
    def __init__(self, data_root, clip_pred_root):
        """
        Args:
            data_root: 指向 sequences 的父目录或 sequences 目录本身
            clip_pred_root: MaskCLIP 2D 预测结果存放目录
        """
        self.data_root = Path(data_root)
        
        # 自动修正路径
        if self.data_root.name != "sequences":
            self.data_root = self.data_root / "sequences"
            
        if not self.data_root.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.data_root}")

        self.clip_pred_root = Path(clip_pred_root)
        self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    def load_calib(self, calib_file):
        """
        【核心修复】读取标定参数，并自动将 Tr 矩阵补全为 4x4
        """
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if ':' not in line: continue
                key, value = line.split(':', 1)
                key = key.strip()
                data = np.array([float(x) for x in value.split()])
                
                # 投影矩阵 P0-P3 保持 3x4
                if key.startswith('P'):
                    calib[key] = data.reshape(3, 4)
                
                # 变换矩阵 Tr 需要补全为 4x4
                elif key.startswith('Tr'):
                    matrix_3x4 = data.reshape(3, 4)
                    matrix_4x4 = np.vstack([matrix_3x4, [0, 0, 0, 1]]) # 补上最后一行
                    calib[key] = matrix_4x4
                    
        return calib

    def project_lidar_to_image(self, points, calib):
        """将 3D 点投影到 2D 图像平面"""
        # 扩展为齐次坐标 [N, 4]
        points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        
        # 变换矩阵 T = P2 (3x4) * Tr (4x4) -> (3x4)
        # 现在维度匹配了：(3,4) @ (4,4) = (3,4)
        trans_matrix = calib['P2'] @ calib['Tr']
        
        points_2d = trans_matrix @ points_hom.T
        points_2d = points_2d.T
        
        # 归一化深度
        mask = points_2d[:, 2] > 0 
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        return points_2d, mask

    def load_maskclip_pred(self, seq, frame_id):
        """读取真实的 2D MaskCLIP 预测"""
        # 优先尝试带 sequences 结构的路径
        pred_path = self.clip_pred_root / "sequences" / seq / f"{frame_id}.npy"
        
        if not pred_path.exists():
            # 尝试直接结构
            pred_path = self.clip_pred_root / seq / f"{frame_id}.npy"
        
        if not pred_path.exists():
            raise FileNotFoundError(f"找不到 MaskCLIP 预测文件: {pred_path}")

        return np.load(pred_path)

    def process(self):
        print(f"数据根目录: {self.data_root}")
        print(f"MaskCLIP 预测目录: {self.clip_pred_root}")
        print("开始生成 3D 投影伪标签...")
        
        for seq in self.sequences:
            print(f"正在处理序列 {seq}...")
            seq_dir = self.data_root / seq
            velodyne_dir = seq_dir / 'velodyne'
            calib_file = seq_dir / 'calib.txt'
            
            # 创建输出目录
            out_dir = seq_dir / 'clip_pseudo'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            calib = self.load_calib(calib_file)
            frames = sorted([f.stem for f in velodyne_dir.glob('*.bin')])
            
            for frame in tqdm(frames):
                target_file = out_dir / f"{frame}.npy"
                if target_file.exists():
                    continue 

                pc_path = velodyne_dir / f"{frame}.bin"
                points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
                
                try:
                    clip_2d_pred = self.load_maskclip_pred(seq, frame)
                except FileNotFoundError as e:
                    print(e)
                    continue

                pts_2d, mask = self.project_lidar_to_image(points, calib)
                
                # 初始化为 255 (ignore)
                clip_labels_3d = np.full(points.shape[0], 255, dtype=np.uint8)
                
                u = np.round(pts_2d[:, 0]).astype(int)
                v = np.round(pts_2d[:, 1]).astype(int)
                H, W = clip_2d_pred.shape
                
                # 过滤视野外的点
                valid_idx = mask & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                
                if valid_idx.sum() > 0:
                    clip_labels_3d[valid_idx] = clip_2d_pred[v[valid_idx], u[valid_idx]]
                
                np.save(str(target_file), clip_labels_3d)

if __name__ == '__main__':
    # ================= 配置区域 =================
    ROOT_DIR = "/media/dell/wcjdata/nzz/PointCT-main/dataset/SemanticKITTI/"
    CLIP_PRED_DIR = "/media/dell/wcjdata/nzz/PointCT-main/dataset/MaskCLIP_Results/" 
    # ===========================================
    
    if not os.path.exists(CLIP_PRED_DIR):
        print(f"Error: MaskCLIP 预测目录不存在: {CLIP_PRED_DIR}")
    else:
        processor = SemanticKITTIPreprocessor(ROOT_DIR, CLIP_PRED_DIR)
        processor.process()
