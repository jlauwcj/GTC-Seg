import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from util import config
from util.s3dis import S3DIS
from util.scannet import Scannetv2
from util.stpls3d import STPLS
from util.semantickitti import SemanticKITTIDataset
from model.semantickitti import hybrid_weak_seg_repro as ModelSemanticKITTI
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util import transform as t
from util.pseudo_label_refinery import PseudoLabelRefinery, HybridLoss, get_optimal_config

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger(log_file):
    """创建日志器"""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    console_handler.setFormatter(logging.Formatter(console_fmt))
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    file_handler.setFormatter(logging.Formatter(file_fmt))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def create_dirs(args):
    """创建必要的目录"""
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'plots'), exist_ok=True)


def plot_metrics(train_loss, val_loss, train_mIoU, val_mIoU, save_path, epoch):
    """绘制训练/验证的损失和mIoU曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Training and Validation Metrics (Epoch {epoch+1})', fontsize=16)
    
    ax1.plot(range(1, len(train_loss)+1), train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    if len(val_loss) > 0:
        ax1.plot(range(1, len(val_loss)+1), val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, len(train_mIoU)+1), train_mIoU, 'b-', label='Train mIoU', linewidth=2, marker='o', markersize=4)
    if len(val_mIoU) > 0:
        ax2.plot(range(1, len(val_mIoU)+1), val_mIoU, 'r-', label='Val mIoU', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('mIoU Curve')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'metrics_curve_epoch_{epoch+1}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'metrics_curve_latest.png'), dpi=300, bbox_inches='tight')
    plt.close()


def custom_collate_fn(batch):
    coords, feats, labels, lengths, cloud_ids, point_idxs, full_sizes = zip(*batch)

    coords = torch.cat(coords, 0).float()
    feats = torch.cat(feats, 0).float()
    labels = torch.cat(labels, 0).long()
    point_idxs = torch.cat([torch.as_tensor(p) for p in point_idxs], 0).long()
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    offset = torch.cumsum(lengths_tensor, dim=0)
    
    cloud_ids = torch.tensor(cloud_ids, dtype=torch.long)
    full_sizes = torch.tensor(full_sizes, dtype=torch.long)

    return coords, feats, labels, offset, lengths_tensor, cloud_ids, point_idxs, full_sizes, None


def collate_fn_semantickitti(batch):
    coords, feats, labels, clip_labels = zip(*batch)
    
    offset, count = [], 0
    for item in coords:
        count += item.shape[0]
        offset.append(count)
    
    coords = torch.cat(coords, 0).float()
    feats = torch.cat(feats, 0).float()
    labels = torch.cat(labels, 0).long()
    clip_labels = torch.cat(clip_labels, 0).long()
    offset = torch.IntTensor(offset)
    
    return coords, feats, labels, offset, None, None, None, None, clip_labels


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    
    create_dirs(args)
    
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name not in ['s3dis','scannet','stpls3d', 'semantickitti']:
        raise NotImplementedError()
    
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0

    if not hasattr(args, 'start_epoch'):
        args.start_epoch = 0
        
    use_amp = (args.data_name != 'scannet')
    
    train_loss_list = []
    train_mIoU_list = []
    val_loss_list = []
    val_mIoU_list = []
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    current_collate_fn = custom_collate_fn

    if args.data_name == 's3dis':
        from model.s3dis import hybrid_weak_seg_repro as Model
        train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, 
                          voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=None, shuffle_index=True, loop=args.loop, labeled_point=args.labeled_point)
        val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=None)

    elif args.data_name == 'scannet':
        from model.scannet import weak_seg_repro as Model
        train_data = Scannetv2(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=None, shuffle_index=True, loop=args.loop, labeled_point=args.labeled_point)
        val_data = Scannetv2(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=None)

    elif args.data_name == 'stpls3d':
        from model.stpls3d import weak_seg_repro as Model
        train_data = STPLS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=None, shuffle_index=True, loop=args.loop, labeled_point=args.labeled_point)
        val_data = STPLS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=None)

    elif args.data_name == 'semantickitti':
        Model = ModelSemanticKITTI
        train_data = SemanticKITTIDataset(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, loop=args.loop, labeled_point=args.labeled_point)
        val_data = SemanticKITTIDataset(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max)
        current_collate_fn = collate_fn_semantickitti

    else:
        raise Exception('dataset not supported: {}'.format(args.data_name))
    
    model = Model(c=args.fea_dim, k=args.classes)
    
    teacher_model = Model(c=args.fea_dim, k=args.classes)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.load_state_dict(model.state_dict())
    
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
       teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
    
    is_weak = '%' in str(args.labeled_point) and float(str(args.labeled_point).strip('%')) < 100
    if '%' in str(args.labeled_point):
        current_ratio = float(str(args.labeled_point).strip('%')) / 100.0
    else:
        current_ratio = 1.0
    
    refinery_cfg = get_optimal_config(
        num_train_clouds=len(train_data.data_list) if hasattr(train_data, 'data_list') else 1000,
        annotation_ratio=current_ratio
    )
    refinery = PseudoLabelRefinery(**refinery_cfg)
    
    criterion = HybridLoss(
        num_classes=args.classes, 
        ignore_index=args.ignore_label, 
        balance_samples=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    if main_process():
        global logger, writer
        log_file = os.path.join(args.save_path, 'logs', 'training.log')
        logger = get_logger(log_file)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        
        if args.data_name == 'semantickitti':
            logger.info("=> Initialized EMA Teacher Model for SemanticKITTI PLR")
            logger.info("=> Class 6/7 excluded from pseudo label generation (parking/other-ground)")
        
        if use_amp:
            logger.info(f"=> AMP (Automatic Mixed Precision) ENABLED for {args.data_name} for faster training.")
        else:
            logger.info(f"=> AMP DISABLED for {args.data_name} (Using standard FP32 with Gradient Clipping).")
        
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu], find_unused_parameters=True if "transformer" in str(args.arch) else False)
        teacher_model = torch.nn.parallel.DistributedDataParallel(
            teacher_model.cuda(), device_ids=[gpu], find_unused_parameters=True if "transformer" in str(args.arch) else False)
    else:
        model = torch.nn.DataParallel(model.cuda())
        teacher_model = torch.nn.DataParallel(teacher_model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            teacher_model.load_state_dict(model.state_dict())
            if main_process(): logger.info("=> loaded weight '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            teacher_model.load_state_dict(model.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_iou = checkpoint['best_iou']
            if main_process(): logger.info("=> loaded checkpoint '{}'".format(args.resume))

    if args.distributed: train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else: train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, 
        drop_last=True, collate_fn=current_collate_fn,
        worker_init_fn=worker_init_fn
    )

    val_loader = None
    if args.evaluate:
        if args.distributed: val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else: val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, 
            pin_memory=True, sampler=val_sampler, collate_fn=current_collate_fn
        )

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(args.start_epoch, args.epochs):
        if main_process():
            print('='*70)
            print(f'Epoch [{epoch+1}/{args.epochs}]')
            print('='*70)
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader, model, teacher_model, criterion, optimizer, epoch, refinery, is_weak, scaler, use_amp
        )
        
        if args.data_name != 'semantickitti':
            refinery.step_epoch()
            refinery.optimize_memory()

        scheduler.step()
        epoch_log = epoch + 1
        
        train_loss_list.append(loss_train)
        train_mIoU_list.append(mIoU_train)

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion, use_amp)
            val_loss_list.append(loss_val)
            val_mIoU_list.append(mIoU_val)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                logger.info('-'*100)
                logger.info(f'Best mIoU so far: {best_iou:.4f}')
        
        if main_process():
            plot_metrics(train_loss_list, val_loss_list, train_mIoU_list, val_mIoU_list, 
                        os.path.join(args.save_path, 'plots'), epoch)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            torch.save({
                'epoch': epoch_log, 
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 
                'best_iou': best_iou, 
                'is_best': is_best
            }, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        plot_metrics(train_loss_list, val_loss_list, train_mIoU_list, val_mIoU_list, 
                    os.path.join(args.save_path, 'plots'), args.epochs-1)


def train(train_loader, model, teacher_model, criterion, optimizer, epoch, refinery, is_weak, scaler, use_amp):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    SKIP_CLASSES = {6, 7}
    
    model.train()
    teacher_model.eval()
    
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    
    for i, (coord, feat, target, offset, lengths, cloud_ids, point_idxs, full_sizes, clip_label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        coord = coord.cuda(non_blocking=True)
        feat = feat.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True).int()
        
        if point_idxs is not None:
            point_idxs = point_idxs.cuda(non_blocking=True)
            lengths = lengths.cuda(non_blocking=True).int()
            cloud_ids = cloud_ids.cuda(non_blocking=True)

        if clip_label is not None:
            clip_label = clip_label.cuda(non_blocking=True)

        if target.shape[-1] == 1:
            target = target[:, 0]

        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            if args.data_name == 'semantickitti':
                output, loss_con = model([coord, feat, offset], clip_labels=clip_label)
            else:
                output = model([coord, feat, offset])
                loss_con = torch.tensor(0.0, device=coord.device)
            
            alpha = 0.0
            pseudo_labels = torch.zeros_like(target).fill_(args.ignore_label)
            
            if is_weak:
                if args.data_name == 'semantickitti':
                    with torch.no_grad():
                        with autocast(enabled=use_amp):
                            teacher_output = teacher_model([coord, feat, offset], clip_labels=None)
                        
                        teacher_prob = torch.softmax(teacher_output.float(), dim=1)
                        max_prob, pseudo_labels = torch.max(teacher_prob, dim=1)
                        
                        if epoch >= 65:
                            pseudo_labels_filtered = torch.zeros_like(pseudo_labels).fill_(args.ignore_label)
                            pseudo_labels_filtered[max_prob >= 0.90] = pseudo_labels[max_prob >= 0.90]
                            
                            for sc in SKIP_CLASSES:
                                pseudo_labels_filtered[pseudo_labels_filtered == sc] = args.ignore_label
                            pseudo_labels = pseudo_labels_filtered
                            alpha = 0.2
                        else:
                            pseudo_labels_filtered = torch.zeros_like(pseudo_labels).fill_(args.ignore_label)
                            for c in range(args.classes):
                                if c in SKIP_CLASSES:
                                    continue
                                c_mask = (pseudo_labels == c)
                                if c_mask.sum() > 0:
                                    c_mean_prob = max_prob[c_mask].mean()
                                    c_thresh = torch.clamp(c_mean_prob * 0.95, min=0.80, max=0.95)
                                    valid_mask = c_mask & (max_prob >= c_thresh)
                                    pseudo_labels_filtered[valid_mask] = c
                            pseudo_labels = pseudo_labels_filtered
                            
                            if epoch < 3:
                                alpha = 0.0
                            else:
                                alpha = (epoch - 3) / 62 * 0.4
                else:
                    with torch.no_grad():
                        refinery.update_global_history(cloud_ids, output, point_idxs, full_sizes, lengths)
                        pseudo_labels, conf_mask = refinery.generate_global_pseudo_labels(cloud_ids, point_idxs, lengths)
                        pseudo_labels[pseudo_labels == -1] = args.ignore_label
                        if epoch < 3: alpha = 0.0
                        elif epoch < 60: alpha = (epoch - 3) / 57 * 0.6
                        else: alpha = 0.6

            loss_main, loss_dict = criterion(output, target, pseudo_labels, alpha=alpha)
            
            if args.data_name == 'semantickitti':
                loss = loss_main + loss_con
            else:
                loss = loss_main
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        step_skipped = (scale_after < scale_before)

        if is_weak and args.data_name == 'semantickitti' and not step_skipped:
            ema_alpha_val = 0.99
            with torch.no_grad():
                for student_param, teacher_param in zip(model.parameters(), teacher_model.parameters()):
                    teacher_param.data = ema_alpha_val * teacher_param.data + (1 - ema_alpha_val) * student_param.data

        output = output.max(1)[1]
        loss_val = loss.float().item()
        n = coord.size(0)
        
        if args.multiprocessing_distributed:
            reduced_loss = loss.clone().float()
            reduced_loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(reduced_loss), dist.all_reduce(count)
            n = count.item()
            loss_val = (reduced_loss / n).item()
        
        intersection, union, target_val = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_val)
        intersection, union, target_val = intersection.cpu().numpy(), union.cpu().numpy(), target_val.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_val)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss_val, n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            log_str = ('Epoch: [{}/{}][{}/{}] '
                       'Data {data_time.val:.3f} '
                       'Batch {batch_time.val:.3f} '
                       'Remain {remain_time} '
                       'Loss {loss_meter.val:.4f} ').format(
                           epoch+1, args.epochs, i + 1, len(train_loader),
                           batch_time=batch_time, data_time=data_time,
                           remain_time=remain_time, loss_meter=loss_meter)
            
            if args.data_name == 'semantickitti':
                cgcm_ratio = loss_con.item() / (loss_main.item() + 1e-8)
                log_str += '(Main:{:.3f}/Con:{:.3f}/Ratio:{:.3f}) '.format(
                    loss_main.item(),
                    loss_con.item() if hasattr(loss_con, 'item') else loss_con,
                    cgcm_ratio
                )
            log_str += 'Acc {accuracy:.4f}'.format(accuracy=accuracy)
            logger.info(log_str)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            epoch+1, args.epochs, mIoU, mAcc, allAcc))
    
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, use_amp):
    if main_process(): logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    
    for i, (coord, feat, target, offset, lengths, cloud_ids, _, _, clip_label) in enumerate(val_loader):
        coord = coord.cuda(non_blocking=True)
        feat = feat.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True).int()
        
        if target.shape[-1] == 1:
            target = target[:, 0]
        
        with torch.no_grad():
            with autocast(enabled=use_amp):
                if args.data_name == 'semantickitti':
                    output = model([coord, feat, offset], clip_labels=None)
                else:
                    output = model([coord, feat, offset])
            
            output = output.float()
            pseudo_labels = torch.zeros_like(target).fill_(args.ignore_label)
            loss, _ = criterion(output, target, pseudo_labels, alpha=0.0)

        output = output.max(1)[1]
        loss_val = loss.float().item()
        n = coord.size(0)
        
        if args.multiprocessing_distributed:
            reduced_loss = loss.clone().float()
            reduced_loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(reduced_loss), dist.all_reduce(count)
            n = count.item()
            loss_val = (reduced_loss / n).item()

        intersection, union, target_val = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_val)
        intersection, union, target_val = intersection.cpu().numpy(), union.cpu().numpy(), target_val.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_val)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss_val, n)
        batch_time.update(time.time() - end)
        
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] Loss {loss_meter.val:.4f} Accuracy {accuracy:.4f}.'.format(
                            i + 1, len(val_loader), loss_meter=loss_meter, accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
