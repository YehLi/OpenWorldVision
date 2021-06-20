#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tools import MultiCropsTransform, GaussianBlur, setup_logger
from tools import (AverageMeter, ValueMeter, ProgressMeter,
                  save_checkpoint, accuracy, adjust_learning_rate)
import models as pretrain_models
import models.backbones as backbones


model_names = sorted(name for name in backbones.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbones.__dict__[name]))
pretrain_model_names = sorted(name for name in pretrain_models.__dict__
    if callable(pretrain_models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# options for dataset
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--img-size', default=224, type=int,
                    help='path to dataset')
# options for model
parser.add_argument('--model', default='MoCo',
                    choices=pretrain_model_names,
                    help='pretrained model: ' +
                        ' | '.join(pretrain_model_names) +
                        ' (default: MoCo)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--load', default='', type=str,
                    help='path to load checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# optimization hyper-parameter
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# options for logger
parser.add_argument('--work-dirs', default='', type=str, metavar='PATH',
                    help='path to workdirs (default: none)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq', default=10, type=int,
                    metavar='N', help='saving models frequency (default: 10)')
# options for distributed setting
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# options for vanilla-q
parser.add_argument('--q-times', default=1, type=int,
                    help='Multiple query times.')
# options for data augmentations
parser.add_argument('--colorjitter-scale', default=1.0, type=float,
                    help='Multiple query times.')
parser.add_argument('--crop-max', default=1.0, type=float,
                    help='Multiple query times.')
parser.add_argument('--crop-min', default=0.2, type=float,
                    help='Multiple query times.')
# seed
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
# fp16
parser.add_argument('--use-fp16', action='store_true',
                    help='use fp16')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    os.makedirs(args.work_dirs, exist_ok=True)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # set up logger
    logger = setup_logger(output=args.work_dirs, distributed_rank=dist.get_rank(), name="moco")

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        logger.info = print_pass

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    logger.info("=> hyper-parameters:\n{}".format(args))

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = pretrain_models.__dict__[args.model](
        backbones.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    logger.info(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally load from a checkpoint
    if args.load:
        if os.path.isfile(args.load):
            logger.info("=> loading checkpoint '{}'".format(args.load))
            if args.gpu is None:
                checkpoint = torch.load(args.load)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.load, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}'".format(args.load))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.load))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        s = args.colorjitter_scale
        augmentation = [
            transforms.RandomResizedCrop(args.img_size, scale=(args.crop_min, args.crop_max)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        MultiCropsTransform(
            transforms.Compose(augmentation),
            times=args.q_times+1))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.end_epochs is None:
        args.end_epochs = args.epochs

    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    for epoch in range(args.start_epoch, args.end_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args, use_cos=args.cos)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, logger, scaler)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.save_freq != 0 and (epoch + 1) < args.epochs - 5:
                continue
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(
                args.work_dirs, 'checkpoint_{:04d}.pth.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch, args, logger=None, scaler=None):
    eta = AverageMeter('eta', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    top1 = AverageMeter('acc@1', ':6.2f')
    top5 = AverageMeter('acc@5', ':6.2f')
    lr = ValueMeter('lr', ':.4e')
    batch_time = AverageMeter('time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    progress_infos = [eta, losses, top1, top5, lr, batch_time, data_time]
    progress = ProgressMeter(
        len(train_loader),
        progress_infos,
        prefix="Epoch: [{}]".format(epoch),
        logger=logger)

    # switch to train mode
    model.train()
    sample_number = len(train_loader)
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = [img.cuda(args.gpu, non_blocking=True) for img in images]
        im_q = torch.cat(images[:-1], dim=0)
        im_k = images[-1]
        assert im_q.size(0) // im_k.size(0) == (args.q_times)

        # compute output
        if args.use_fp16:
            with torch.cuda.amp.autocast():
                output, target = model(im_q=im_q, im_k=im_k)
                loss = criterion(output, target)
        else:
            output, target = model(im_q=im_q, im_k=im_k)
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(acc1[0], output.size(0))
        top5.update(acc5[0], output.size(0))
        lr.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        estimated_left_time = batch_time.val * ((args.end_epochs - epoch) * sample_number - i) / 3600.0
        eta.update(estimated_left_time)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()
