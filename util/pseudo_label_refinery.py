import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MemoryEfficientPseudoLabelRefinery:
  
    def __init__(
        self,
        num_classes=13,
        history_size=5,
        confidence_threshold=0.90,
        use_adaptive_threshold=False,
        min_samples_per_class=10,
        temperature=1.0,
        ema_decay=0.999,
        max_cache_clouds=2000,  
        use_cpu_storage=True,
        use_fp16=True
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
        self.global_history = {}
        self.access_count = {}
        self.current_epoch = 0
        self.adaptive_threshold = confidence_threshold
        logger.info(f"MemoryEfficientPseudoLabelRefinery initialized")
        logger.info(f"  Storage: {'CPU' if use_cpu_storage else 'GPU'}")
        logger.info(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
        logger.info(f"  Confidence Threshold: {self.confidence_threshold}")
        logger.info(f"  Max cache: {max_cache_clouds} clouds")
    
    def _maybe_cleanup_cache(self):
        if len(self.global_history) <= self.max_cache_clouds:
            return
        sorted_clouds = sorted(self.access_count.items(), key=lambda x: x[1])
        num_remove = max(1, len(sorted_clouds) // 5)
        clouds_to_remove = [cloud_id for cloud_id, _ in sorted_clouds[:num_remove]]
        
        for cloud_id in clouds_to_remove:
            if cloud_id in self.global_history:
                del self.global_history[cloud_id]
            if cloud_id in self.access_count:
                del self.access_count[cloud_id]
        
        logger.info(f"Cleaned cache: removed {len(clouds_to_remove)} clouds")
    
    def update_global_history(self, cloud_ids, logits, point_idxs, full_sizes, lengths):
        actual_num_classes = logits.size(1)
        if self.num_classes != actual_num_classes:
            logger.info(f"=> Auto-adjusting PseudoLabelRefinery num_classes: {self.num_classes} -> {actual_num_classes}")
            self.num_classes = actual_num_classes
            self.global_history.clear()
            self.access_count.clear()

        batch_size = cloud_ids.size(0)
        cloud_ids_list = cloud_ids.cpu().tolist()
        full_sizes_list = full_sizes.cpu().tolist()
        lengths_list = lengths.cpu().tolist()
        offset = [0]
        for i in range(len(lengths_list) - 1):
            offset.append(offset[-1] + lengths_list[i])
        
        dtype = torch.float16 if self.use_fp16 else torch.float32
        ema_momentum = 0.7  
        for i in range(batch_size):
            cloud_id = cloud_ids_list[i]
            full_size = full_sizes_list[i]
            start_idx = offset[i]
            end_idx = offset[i] + lengths_list[i]
            if start_idx >= logits.size(0) or end_idx > logits.size(0):
                continue
            batch_logits = torch.clamp(logits[start_idx:end_idx].detach().float(), min=-20.0, max=20.0).cpu()
            batch_point_idxs = point_idxs[start_idx:end_idx].cpu().long()
            if batch_point_idxs.numel() == 0 or batch_point_idxs.min() < 0 or batch_point_idxs.max() >= full_size:
                continue

            if cloud_id not in self.global_history or self.global_history[cloud_id]['logits'].size(1) != self.num_classes:
                self.global_history[cloud_id] = {
                    'logits': torch.zeros(full_size, self.num_classes, dtype=dtype),
                    'counts': torch.zeros(full_size, dtype=torch.int16)
                }
                self.access_count[cloud_id] = 0
            self.access_count[cloud_id] += 1
            history = self.global_history[cloud_id]
            try:
                prev_counts = history['counts'][batch_point_idxs]
                first_visit_mask = (prev_counts == 0)
                repeat_visit_mask = ~first_visit_mask
                
                if first_visit_mask.any():
                    first_idxs = batch_point_idxs[first_visit_mask]
                    history['logits'][first_idxs] = batch_logits[first_visit_mask].to(dtype)
                
                if repeat_visit_mask.any():
                    repeat_idxs = batch_point_idxs[repeat_visit_mask]
                    old_logits = history['logits'][repeat_idxs].float()
                    new_logits = ema_momentum * old_logits + (1.0 - ema_momentum) * batch_logits[repeat_visit_mask]
                    history['logits'][repeat_idxs] = new_logits.to(dtype)
                
                history['counts'][batch_point_idxs] = torch.clamp(prev_counts + 1, max=30000)
            except Exception as e:
                logger.warning(f"History update failed for cloud {cloud_id}: {e}")
        self._maybe_cleanup_cache()
    
    def generate_global_pseudo_labels(self, cloud_ids, point_idxs, lengths):
        batch_size = cloud_ids.size(0)
        total_points = point_idxs.size(0)
        device = point_idxs.device
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
            
            batch_point_idxs = point_idxs[start_idx:end_idx].cpu().long()
            
            if batch_point_idxs.numel() == 0 or batch_point_idxs.min() < 0 or batch_point_idxs.max() >= history['logits'].size(0):
                continue
            
            try:
                batch_logits = torch.index_select(history['logits'], 0, batch_point_idxs).float()
                batch_counts = torch.index_select(history['counts'], 0, batch_point_idxs)
            except Exception:
                continue
            
            valid_mask = batch_counts > 0
            if valid_mask.sum() == 0:
                continue
            probs = F.softmax(batch_logits / self.temperature, dim=1)
            max_probs, preds = probs.max(dim=1)
            
            confident = (max_probs >= self.adaptive_threshold) & valid_mask

            if confident.sum() > 0:
                pseudo_labels[start_idx:end_idx][confident] = preds[confident].to(device)
                conf_mask[start_idx:end_idx] = confident.to(device)
        
        return pseudo_labels, conf_mask
    
    def update_adaptive_threshold(self, logits, targets, percentile=90.0):
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
                
                self.adaptive_threshold = max(0.85, min(0.95, self.adaptive_threshold))
        except Exception:
            pass
    
    def compute_pseudo_label_quality(self, pseudo_labels, targets, conf_mask):
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
        except Exception:
            return {'accuracy': 0.0, 'coverage': 0.0, 'num_pseudo': 0}
    
    def step_epoch(self):
        self.current_epoch += 1
    
    def optimize_memory(self):
        self._maybe_cleanup_cache()
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

PseudoLabelRefinery = MemoryEfficientPseudoLabelRefinery


class HybridLoss(nn.Module):
    
    def __init__(self, num_classes, ignore_index=-1, balance_samples=True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.balance_samples = balance_samples
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    
    def forward(self, logits, targets, pseudo_labels=None, alpha=0.0):
        loss_gt = self.ce_loss(logits, targets)
        
        if pseudo_labels is None or alpha <= 0:
            return loss_gt, {
                'loss_gt': loss_gt.item(),
                'loss_pseudo': 0.0,
                'num_pseudo': 0
            }
        
        try:
            valid_gt = (targets != self.ignore_index) & (targets != -1)
            valid_pseudo = (pseudo_labels != self.ignore_index) & (pseudo_labels != -1) & (~valid_gt)
            
            if valid_pseudo.sum() == 0:
                return loss_gt, {
                    'loss_gt': loss_gt.item(),
                    'loss_pseudo': 0.0,
                    'num_pseudo': 0
                }
            loss_pseudo = F.cross_entropy(
                logits[valid_pseudo], 
                pseudo_labels[valid_pseudo],
                reduction='mean'
            )
            total_loss = (1 - alpha) * loss_gt + alpha * loss_pseudo
            
            return total_loss, {
                'loss_gt': loss_gt.item(),
                'loss_pseudo': loss_pseudo.item(),
                'num_pseudo': valid_pseudo.sum().item()
            }
        except Exception:
            return loss_gt, {
                'loss_gt': loss_gt.item(),
                'loss_pseudo': 0.0,
                'num_pseudo': 0
            }


def get_optimal_config(num_train_clouds, annotation_ratio, num_classes=13):
    config = {
        'num_classes': num_classes,
        'history_size': 5,
        'confidence_threshold': 0.90,
        'use_adaptive_threshold': False,
        'min_samples_per_class': 10,
        'temperature': 1.0,
        'ema_decay': 0.999,
        'max_cache_clouds': max(2000, num_train_clouds),
        'use_cpu_storage': True,
        'use_fp16': True
    }
    return config
