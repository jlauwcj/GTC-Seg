"""
内存优化版 PseudoLabelRefinery

主要优化：
1. 使用 CPU 存储历史，只在需要时转到 GPU
2. 限制缓存大小
3. 使用 float16 减少内存
4. 动态清理旧数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MemoryEfficientPseudoLabelRefinery:
    """
    内存优化的伪标签精炼模块
    
    关键优化：
    - 历史存储在 CPU（而非 GPU）
    - 使用 float16 降低内存占用
    - 限制最大缓存数量
    - 自动清理策略
    """
    
    def __init__(
        self,
        num_classes=13,
        history_size=5,  # 减少历史大小
        confidence_threshold=0.95,
        use_adaptive_threshold=True,
        min_samples_per_class=10,
        temperature=1.0,
        ema_decay=0.999,
        max_cache_clouds=50,  # 最大缓存点云数
        use_cpu_storage=True,  # 使用 CPU 存储
        use_fp16=True  # 使用半精度
    ):
        self.num_classes = num_classes
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.min_samples_per_class = min_samples_per_class
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.max_cache_clouds = max_cache_clouds
        self.use_cpu_storage = use_cpu_storage
        self.use_fp16 = use_fp16
        
        # 全局历史缓冲区 - 存储在 CPU
        self.global_history = {}
        
        # 访问计数（用于 LRU 淘汰）
        self.access_count = {}
        
        # 当前epoch
        self.current_epoch = 0
        
        # 自适应阈值
        self.adaptive_threshold = confidence_threshold
        
        logger.info(f"MemoryEfficientPseudoLabelRefinery initialized")
        logger.info(f"  Storage: {'CPU' if use_cpu_storage else 'GPU'}")
        logger.info(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
        logger.info(f"  Max cache: {max_cache_clouds} clouds")
    
    def _maybe_cleanup_cache(self):
        """自动清理缓存"""
        if len(self.global_history) <= self.max_cache_clouds:
            return
        
        # LRU 淘汰：删除访问次数最少的
        sorted_clouds = sorted(self.access_count.items(), key=lambda x: x[1])
        clouds_to_remove = [cloud_id for cloud_id, _ in sorted_clouds[:len(sorted_clouds)//2]]
        
        for cloud_id in clouds_to_remove:
            if cloud_id in self.global_history:
                del self.global_history[cloud_id]
            if cloud_id in self.access_count:
                del self.access_count[cloud_id]
        
        logger.info(f"Cleaned cache: removed {len(clouds_to_remove)} clouds")
    
    def update_global_history(self, cloud_ids, logits, point_idxs, full_sizes, lengths):
        """
        更新全局历史 - 内存优化版
        
        关键优化：
        1. 在 CPU 上累积，避免 GPU 内存爆炸
        2. 使用 FP16 存储
        3. 批量操作减少传输
        """
        batch_size = cloud_ids.size(0)
        
        # 转到 CPU
        cloud_ids_list = cloud_ids.cpu().tolist()
        full_sizes_list = full_sizes.cpu().tolist()
        lengths_list = lengths.cpu().tolist()
        
        # 计算偏移
        offset = [0]
        for i in range(len(lengths_list) - 1):
            offset.append(offset[-1] + lengths_list[i])
        
        # 逐个处理
        for i in range(batch_size):
            cloud_id = cloud_ids_list[i]
            full_size = full_sizes_list[i]
            
            start_idx = offset[i]
            end_idx = offset[i] + lengths_list[i]
            
            # 边界检查
            if start_idx >= logits.size(0) or end_idx > logits.size(0):
                continue
            
            # [关键] 立即转到 CPU 并转 FP16（如果启用）
            batch_logits = logits[start_idx:end_idx].detach().cpu()
            batch_point_idxs = point_idxs[start_idx:end_idx].cpu()
            
            if self.use_fp16:
                batch_logits = batch_logits.half()
            
            # 验证索引
            if batch_point_idxs.min() < 0 or batch_point_idxs.max() >= full_size:
                continue
            
            # 初始化历史
            if cloud_id not in self.global_history:
                dtype = torch.float16 if self.use_fp16 else torch.float32
                self.global_history[cloud_id] = {
                    'logits': torch.zeros(full_size, self.num_classes, dtype=dtype),
                    'counts': torch.zeros(full_size, dtype=dtype)
                }
                self.access_count[cloud_id] = 0
            
            # 更新访问计数
            self.access_count[cloud_id] += 1
            
            history = self.global_history[cloud_id]
            
            # [关键] 在 CPU 上累积
            try:
                expanded_idxs = batch_point_idxs.unsqueeze(1).expand(-1, self.num_classes)
                history['logits'].scatter_add_(0, expanded_idxs, batch_logits)
                history['counts'].scatter_add_(0, batch_point_idxs, 
                    torch.ones_like(batch_point_idxs, dtype=history['counts'].dtype))
            except Exception as e:
                logger.warning(f"Scatter failed for cloud {cloud_id}: {e}")
                # 降级方案
                for j in range(batch_point_idxs.size(0)):
                    idx = batch_point_idxs[j].item()
                    if 0 <= idx < full_size:
                        history['logits'][idx] += batch_logits[j]
                        history['counts'][idx] += 1.0
        
        # 定期清理
        self._maybe_cleanup_cache()
    
    def generate_global_pseudo_labels(self, cloud_ids, point_idxs, lengths):
        """
        生成伪标签 - 内存优化版
        
        关键优化：
        1. 只把需要的数据传到 GPU
        2. 处理完立即释放
        """
        batch_size = cloud_ids.size(0)
        total_points = point_idxs.size(0)
        device = point_idxs.device
        
        # 初始化输出（在 GPU）
        pseudo_labels = torch.full((total_points,), -1, dtype=torch.long, device=device)
        conf_mask = torch.zeros(total_points, dtype=torch.bool, device=device)
        
        cloud_ids_list = cloud_ids.cpu().tolist()
        lengths_list = lengths.cpu().tolist()
        
        offset = [0]
        for i in range(len(lengths_list) - 1):
            offset.append(offset[-1] + lengths_list[i])
        
        for i in range(batch_size):
            cloud_id = cloud_ids_list[i]
            
            if cloud_id not in self.global_history:
                continue
            
            history = self.global_history[cloud_id]
            
            start_idx = offset[i]
            end_idx = offset[i] + lengths_list[i]
            
            if start_idx >= total_points or end_idx > total_points:
                continue
            
            batch_point_idxs = point_idxs[start_idx:end_idx].cpu()  # 转到 CPU
            
            if batch_point_idxs.min() < 0 or batch_point_idxs.max() >= history['logits'].size(0):
                continue
            
            # [关键] 在 CPU 上计算，只把结果传回 GPU
            try:
                batch_logits = torch.index_select(history['logits'], 0, batch_point_idxs)
                batch_counts = torch.index_select(history['counts'], 0, batch_point_idxs)
            except:
                continue
            
            valid_mask = batch_counts > 0
            if valid_mask.sum() == 0:
                continue
            
            # 计算平均（在 CPU）
            avg_logits = batch_logits.clone()
            avg_logits[valid_mask] = batch_logits[valid_mask] / batch_counts[valid_mask].unsqueeze(1)
            
            # 转回 FP32（如果需要）
            if self.use_fp16:
                avg_logits = avg_logits.float()
            
            # Softmax（在 CPU）
            probs = F.softmax(avg_logits / self.temperature, dim=1)
            max_probs, preds = probs.max(dim=1)
            
            confident = (max_probs > self.adaptive_threshold) & valid_mask
            
            # [关键] 只把结果传到 GPU
            if confident.sum() > 0:
                pseudo_labels[start_idx:end_idx][confident] = preds[confident].to(device)
                conf_mask[start_idx:end_idx] = confident.to(device)
        
        return pseudo_labels, conf_mask
    
    def update_adaptive_threshold(self, logits, targets, percentile=90.0):
        """更新自适应阈值"""
        if not self.use_adaptive_threshold:
            return
        
        try:
            with torch.no_grad():
                probs = F.softmax(logits / self.temperature, dim=1)
                max_probs, preds = probs.max(dim=1)
                
                valid_mask = (targets != -1) & (targets != 255)
                if valid_mask.sum() == 0:
                    return
                
                correct_mask = valid_mask & (preds == targets)
                if correct_mask.sum() == 0:
                    return
                
                correct_probs = max_probs[correct_mask]
                threshold = torch.quantile(correct_probs, percentile / 100.0)
                
                self.adaptive_threshold = (
                    self.ema_decay * self.adaptive_threshold + 
                    (1 - self.ema_decay) * threshold.item()
                )
                
                self.adaptive_threshold = max(0.5, min(0.99, self.adaptive_threshold))
        except:
            pass
    
    def compute_pseudo_label_quality(self, pseudo_labels, targets, conf_mask):
        """计算伪标签质量"""
        try:
            valid_gt = (targets != -1) & (targets != 255)
            confident_with_gt = conf_mask & valid_gt
            
            if confident_with_gt.sum() == 0:
                return {'accuracy': 0.0, 'coverage': 0.0, 'num_pseudo': 0}
            
            correct = (pseudo_labels[confident_with_gt] == targets[confident_with_gt])
            accuracy = correct.float().mean().item()
            coverage = confident_with_gt.sum().item() / max(valid_gt.sum().item(), 1)
            
            return {
                'accuracy': accuracy,
                'coverage': coverage,
                'num_pseudo': conf_mask.sum().item()
            }
        except:
            return {'accuracy': 0.0, 'coverage': 0.0, 'num_pseudo': 0}
    
    def step_epoch(self):
        """Epoch结束"""
        self.current_epoch += 1
    
    def optimize_memory(self):
        """优化内存"""
        self._maybe_cleanup_cache()
        
        # 强制清理
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 保持原接口兼容
PseudoLabelRefinery = MemoryEfficientPseudoLabelRefinery


class HybridLoss(nn.Module):
    """混合损失函数"""
    
    def __init__(self, num_classes, ignore_index=-1, balance_samples=True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.balance_samples = balance_samples
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    
    def forward(self, logits, targets, pseudo_labels=None, alpha=0.0):
        """前向传播"""
        # GT 损失
        loss_gt = self.ce_loss(logits, targets)
        
        if pseudo_labels is None or alpha <= 0:
            return loss_gt, {
                'loss_gt': loss_gt.item(),
                'loss_pseudo': 0.0,
                'num_pseudo': 0
            }
        
        try:
            valid_pseudo = (pseudo_labels != self.ignore_index) & (pseudo_labels != -1)
            
            if valid_pseudo.sum() == 0:
                return loss_gt, {
                    'loss_gt': loss_gt.item(),
                    'loss_pseudo': 0.0,
                    'num_pseudo': 0
                }
            
            # 伪标签损失
            loss_pseudo = F.cross_entropy(
                logits[valid_pseudo], 
                pseudo_labels[valid_pseudo],
                reduction='mean'
            )
            
            # 混合
            total_loss = (1 - alpha) * loss_gt + alpha * loss_pseudo
            
            return total_loss, {
                'loss_gt': loss_gt.item(),
                'loss_pseudo': loss_pseudo.item(),
                'num_pseudo': valid_pseudo.sum().item()
            }
        except:
            return loss_gt, {
                'loss_gt': loss_gt.item(),
                'loss_pseudo': 0.0,
                'num_pseudo': 0
            }


def get_optimal_config(num_train_clouds, annotation_ratio):
    """获取优化配置"""
    config = {
        'num_classes': 13,
        'history_size': 5,  # 减少
        'confidence_threshold': 0.95,
        'use_adaptive_threshold': True,
        'min_samples_per_class': 10,
        'temperature': 1.0,
        'ema_decay': 0.999,
        'max_cache_clouds': min(50, num_train_clouds),  # 限制缓存
        'use_cpu_storage': True,  # 使用 CPU 存储
        'use_fp16': True  # 使用半精度
    }
    
    if annotation_ratio < 0.01:
        config['confidence_threshold'] = 0.98
        config['history_size'] = 8
    elif annotation_ratio < 0.05:
        config['confidence_threshold'] = 0.95
        config['history_size'] = 5
    else:
        config['confidence_threshold'] = 0.90
        config['history_size'] = 3
    
    return config

