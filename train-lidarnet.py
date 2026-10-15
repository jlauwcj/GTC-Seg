import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util import config
from util.s3dis import S3DIS
from util.scannet import Scannetv2
from util.stpls3d import STPLS
from util.lidarnet import Lidarnet
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util import transform as t

from util.pseudo_label_refinery import PseudoLabelRefinery, HybridLoss, get_optimal_config


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


def get_logger(log_dir=None):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        formatter = logging.Formatter(fmt)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


def plot_and_save_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if history['train_epochs']:
        ax1.plot(history['train_epochs'], history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=3)
    if history['val_epochs']:
        ax1.plot(history['val_epochs'], history['val_loss'], label='Val Loss', color='red', marker='s', markersize=3)
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    if history['train_epochs']:
        ax2.plot(history['train_epochs'], history['train_miou'], label='Train mIoU', color='blue', marker='o', markersize=3)
    if history['val_epochs']:
        ax2.plot(history['val_epochs'], history['val_miou'], label='Val mIoU', color='red', marker='s', markersize=3)
    ax2.set_title('mIoU Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300)
    plt.close()


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def custom_collate_fn(batch):
    coords, feats, labels, lengths, cloud_ids, point_idxs, full_sizes = zip(*batch)
    coords = torch.cat(coords, 0).float()
    feats = torch.cat(feats, 0).float()
    labels = torch.cat(labels, 0).long()
    
    point_idxs = torch.cat([torch.as_tensor(p) for p in point_idxs], 0).long()
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    offset = torch.cumsum(lengths_tensor, dim=0).int()
    
    cloud_ids = torch.tensor(cloud_ids, dtype=torch.long)
    full_sizes = torch.tensor(full_sizes, dtype=torch.long)
    return coords, feats, labels, offset, lengths_tensor, cloud_ids, point_idxs, full_sizes


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

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

    if args.data_name not in ['s3dis','scannet','stpls3d', 'lidarnet']:
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
    
    history = {'train_epochs': [], 'train_loss': [], 'train_miou': [], 
               'val_epochs': [], 'val_loss': [], 'val_miou': []}
               
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.data_name == 's3dis':
        from model.s3dis import weak_seg_repro as Model
        train_transform = t.Compose(
        [
            t.RandomScale([0.9, 1.1]), 
            t.ChromaticAutoContrast(), 
            t.ChromaticTranslation(), 
            t.ChromaticJitter(), 
            t.HueSaturationTranslation()
        ])
        train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop,labeled_point=args.labeled_point)
        val_transform = None
        val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)

    elif args.data_name == 'scannet':
        from model.scannet import weak_seg_repro as Model
        train_transform = t.Compose([
            t.RandomScale([0.9, 1.1]),
            t.ChromaticAutoContrast_SN(), 
            t.ChromaticTranslation_SN(), 
            t.ChromaticJitter_SN()
        ])
        train_data = Scannetv2(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop,labeled_point=args.labeled_point)
        val_transform = None
        val_data = Scannetv2(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)

    elif args.data_name == 'stpls3d':
        from model.stpls3d import weak_seg_repro as Model
        train_transform = t.Compose(
        [
            t.RandomScale([0.9, 1.1]), 
            t.ChromaticAutoContrast(), 
            t.ChromaticTranslation(), 
            t.ChromaticJitter(), 
            t.HueSaturationTranslation()
        ])
        train_data = STPLS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop,labeled_point=args.labeled_point)
        val_transform = None
        val_data = STPLS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)

    elif args.data_name == 'lidarnet':
        from model.lidarnet import weak_seg_repro as Model
        train_transform = t.Compose([
            t.RandomScale([0.9, 1.1]),
            t.ChromaticAutoContrast_SN(), 
            t.ChromaticTranslation_SN(), 
            t.ChromaticJitter_SN()
        ])
        train_data = Lidarnet(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop, labeled_point=args.labeled_point, ignore_label=args.ignore_label)
        val_transform = None
        val_data = Lidarnet(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform, ignore_label=args.ignore_label)

    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    
    model = Model(c=args.fea_dim, k=args.classes)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
       
    is_weak = '%' in str(args.labeled_point) and float(str(args.labeled_point).strip('%')) < 100
    current_ratio = float(str(args.labeled_point).strip('%')) / 100.0 if '%' in str(args.labeled_point) else 1.0
    refinery_cfg = get_optimal_config(len(train_data), current_ratio)
    
    refinery_cfg['num_classes'] = args.classes 
    
    refinery = PseudoLabelRefinery(**refinery_cfg)
    criterion = HybridLoss(num_classes=args.classes, ignore_index=args.ignore_label).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_iou = checkpoint['best_iou']
            
            if 'history' in checkpoint:
                history = checkpoint['history']
                
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
            
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=custom_collate_fn)

    val_loader = None
    if args.evaluate:
        val_transform = None

        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=custom_collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        print('======================================')
        print(args.save_path + '/model/model_last.pth')
        print('======================================')
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, refinery, is_weak)
        
        refinery.step_epoch()
        refinery.optimize_memory()
        scheduler.step()
        epoch_log = epoch + 1
        
        history['train_epochs'].append(epoch_log)
        history['train_loss'].append(loss_train)
        history['train_miou'].append(mIoU_train)
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            history['val_epochs'].append(epoch_log)
            history['val_loss'].append(loss_val)
            history['val_miou'].append(mIoU_val)
            
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                print('-'*100)
                print(best_iou)
                print('-'*100)
                
        if main_process():
            plot_and_save_curves(history, args.save_path)
            
            if (epoch_log % args.save_freq == 0):
                filename = args.save_path + '/model/model_last.pth'
                logger.info('Saving checkpoint to: ' + filename)
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best,
                            'history': history}, filename)
                if is_best:
                    logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                    shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch, refinery, is_weak):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    pseudo_acc_meter = AverageMeter()
    pseudo_cov_meter = AverageMeter()
    pseudo_num_meter = AverageMeter()
    
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    
    for i, (coord, feat, target, offset, lengths, cloud_ids, point_idxs, full_sizes) in enumerate(train_loader):
        coord, feat, target = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True).int()
        lengths = lengths.cuda(non_blocking=True).int()
        point_idxs = point_idxs.cuda(non_blocking=True)
        
        data_time.update(time.time() - end)
        
        if target.shape[-1] == 1:
            target = target[:, 0]
            
        optimizer.zero_grad()
        
        output = model([coord, feat, offset])
        
        alpha = 0.0
        pseudo_labels = torch.zeros_like(target).fill_(args.ignore_label)
        
        if is_weak:
            with torch.no_grad():
                refinery.update_global_history(cloud_ids, output.detach(), point_idxs, full_sizes, lengths)
                pseudo_labels, conf_mask = refinery.generate_global_pseudo_labels(cloud_ids, point_idxs, lengths)
                pseudo_labels[pseudo_labels == -1] = args.ignore_label
                alpha = min(0.6, max(0.0, (epoch - 3) / 57 * 0.6)) if epoch >= 3 else 0.0
        
        loss, loss_dict = criterion(output, target, pseudo_labels, alpha=alpha)
        
        if is_weak and isinstance(loss_dict, dict) and epoch >= 3:
            p_acc = loss_dict.get('pseudo_acc', 0.0)
            p_cov = loss_dict.get('coverage', 0.0)
            p_num = loss_dict.get('num_pseudo', 0.0)
            pseudo_acc_meter.update(p_acc)
            pseudo_cov_meter.update(p_cov)
            pseudo_num_meter.update(p_num)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        output = output.detach().max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        
        intersection, union, target_val = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_val)
        intersection, union, target_val = intersection.cpu().numpy(), union.cpu().numpy(), target_val.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_val)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            current_pseudo_num = loss_dict.get('num_pseudo', 0) if isinstance(loss_dict, dict) else 0
            loss_g = loss_dict.get('loss_ce', 0.0) if isinstance(loss_dict, dict) else 0.0
            loss_p = loss_dict.get('loss_pseudo', 0.0) if isinstance(loss_dict, dict) else 0.0

            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} '
                        'Batch {batch_time.val:.3f} '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} (G:{loss_g:.3f}/P:{loss_p:.3f}) '
                        'Acc {accuracy:.4f} '
                        'Alpha {alpha:.3f} '
                        'Pseudo_Num {pseudo_num}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          loss_g=loss_g,
                                                          loss_p=loss_p,
                                                          accuracy=accuracy,
                                                          alpha=alpha,
                                                          pseudo_num=current_pseudo_num))
        
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target_val + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    if is_weak and epoch % 5 == 0 and main_process():
        refinery.update_adaptive_threshold(output.detach(), target, percentile=90.0)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
        
        if is_weak and epoch >= 3:
            logger.info('======================================================')
            logger.info('PCL Quality Summary (Epoch {}):'.format(epoch + 1))
            logger.info('  Avg Accuracy: {:.3f}'.format(pseudo_acc_meter.avg))
            logger.info('  Avg Coverage: {:.3f}'.format(pseudo_cov_meter.avg))
            logger.info('  Avg Num Pseudo: {:.0f}'.format(pseudo_num_meter.avg))
            logger.info('======================================================')

    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    
    for i, (coord, feat, target, offset, lengths, cloud_ids, point_idxs, full_sizes) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True).int()
        
        if target.shape[-1] == 1:
            target = target[:, 0]
            
        with torch.no_grad():
            output = model([coord, feat, offset])
        
        loss, _ = criterion(output, target, alpha=0.0)

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target_val = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_val)
        intersection, union, target_val = intersection.cpu().numpy(), union.cpu().numpy(), target_val.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_val)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

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
