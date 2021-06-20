# https://github.com/rwightman/pytorch-image-models

import argparse
import time
import yaml
import os
import sys
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from config import cfg, resolve_data_config, pop_unused_value
from datasets import Dataset, DatasetTxtNpz, DatasetTxt, DatasetTxtGAN, create_loader_index, resolve_data_config, \
    Mixup, FastCollateMixup, AugMixDataset
from models import create_model, resume_checkpoint, convert_splitbn_model, load_checkpoint, model_parameters
from utils import *
from loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from optim import create_optimizer
from scheduler import create_scheduler
from utils import ApexScaler, NativeScaler
from utils.logger import logger_info, setup_default_logging
import utils.distributed as dist
from utils.flops_counter import get_model_complexity_info
# from evaler.owv_evaler import OpenWorldVisionEvaler
from evaler.bert_openset_evaler import BertOpensetEvaler
from evaler.bert_offline_openset_evaler import BertOfflineOpensetEvaler
from models.registry import is_model, is_model_in_modules, model_entrypoint

if cfg.amp == True:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

torch.backends.cudnn.benchmark = True


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Imagenet Model')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def setup_model():
    model = create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.drop,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=cfg.model.drop_path if 'drop_path' in cfg.model else None,
        drop_block_rate=cfg.model.drop_block if 'drop_block' in cfg.model else None,
        global_pool=cfg.model.gp,
        bn_tf=cfg.BN.bn_tf,
        bn_momentum=cfg.BN.bn_momentum if 'bn_momentum' in cfg.BN else None,
        bn_eps=cfg.BN.bn_eps if 'bn_eps' in cfg.BN else None,
        checkpoint_path=cfg.model.initial_checkpoint)
    data_config = resolve_data_config(cfg, model=model)

    flops_count, params_count = get_model_complexity_info(model, data_config['input_size'], as_strings=True,
                                                          print_per_layer_stat=False, verbose=False)
    logger_info('Model %s created, flops_count: %s, param count: %s' % (cfg.model.name, flops_count, params_count))

    if cfg.BN.split_bn:
        assert cfg.augmentation.aug_splits > 1 or cfg.augmentation.resplit
        model = convert_splitbn_model(model, max(cfg.augmentation.aug_splits, 2))
    model.cuda()

    model_name = 'bert'
    model_args = dict(
        layer_num=cfg.bert.layers,
        hidden_size=cfg.bert.dim,
        intermediate_size=cfg.bert.intermediate_dim,
        num_attention_heads=cfg.bert.head_num,
        dropout=cfg.bert.drop,
        att_dropout=cfg.bert.att_drop,
        num_classes=cfg.model.num_classes,
        num_berts=cfg.bert.bert_num)
    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        bert_model = create_fn(**model_args)
    bert_model.cuda()

    return model, bert_model, data_config


def setup_resume(local_rank, model, bert_model, optimizer):
    loss_scaler = None
    if cfg.amp == True:
        bert_model, optimizer = amp.initialize(bert_model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
    else:
        logger_info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if cfg.model.resume:
        load_checkpoint(model, cfg.model.resume, use_ema=True)

    if cfg.bert.resume:
        resume_epoch = resume_checkpoint(
            bert_model, cfg.bert.resume,
            optimizer=None if cfg.model.no_resume_opt else optimizer,
            loss_scaler=None if cfg.model.no_resume_opt else loss_scaler,
            log_info=local_rank == 0)

    if cfg.distributed:
        model = NativeDDP(model, device_ids=[local_rank])
        bert_model = NativeDDP(bert_model, device_ids=[local_rank])

    model_ema = None
    if cfg.model.model_ema == True:
        model_ema = ModelEmaV2(
            unwrap_model(bert_model),
            decay=cfg.model.model_ema_decay,
            device='cpu' if cfg.model.model_ema_force_cpu else None
        )
        if cfg.bert.resume:
            load_checkpoint(model_ema.module, cfg.bert.resume, use_ema=True)

    return model, bert_model, model_ema, optimizer, resume_epoch, loss_scaler


def setup_scheduler(optimizer, resume_epoch):
    lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)
    start_epoch = 0
    if 'start_epoch' in cfg.solver:
        # a specified start_epoch will always override the resume epoch
        start_epoch = cfg.solver.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    return lr_scheduler, start_epoch, num_epochs


def setup_loader(data_config):
    assert os.path.exists(cfg.data_loader.data_path)
    if cfg.data_loader.use_gan_data:
        dataset_train = DatasetTxtGAN(cfg.data_loader.data_path,
                                      cfg.data_loader.train_label,
                                      cfg.data_loader.train_gan_label)
    else:
        dataset_train = DatasetTxtNpz(cfg.data_loader.data_path, cfg.data_loader.train_label)

    collate_fn = None
    mixup_fn = None
    mixup_active = cfg.augmentation.mixup > 0 or cfg.augmentation.cutmix > 0. or len(cfg.augmentation.cutmix_minmax) > 0
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=cfg.augmentation.mixup, cutmix_alpha=cfg.augmentation.cutmix,
            cutmix_minmax=cfg.augmentation.cutmix_minmax,
            prob=cfg.augmentation.mixup_prob, switch_prob=cfg.augmentation.mixup_switch_prob,
            mode=cfg.augmentation.mixup_mode,
            label_smoothing=cfg.loss.smoothing, num_classes=cfg.model.num_classes)
        if cfg.data_loader.prefetcher:
            assert not cfg.augmentation.aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if cfg.augmentation.aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=cfg.augmentation.aug_splits)

    train_interpolation = cfg.augmentation.train_interpolation
    if cfg.augmentation.no_aug:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader_index(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=cfg.data_loader.batch_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=cfg.augmentation.no_aug,
        re_prob=cfg.augmentation.reprob,
        re_mode=cfg.augmentation.remode,
        re_count=cfg.augmentation.recount,
        re_split=cfg.augmentation.resplit,
        scale=cfg.augmentation.scale,
        ratio=cfg.augmentation.ratio,
        hflip=cfg.augmentation.hflip,
        vflip=cfg.augmentation.vflip,
        color_jitter=cfg.augmentation.color_jitter if cfg.augmentation.color_jitter > 0 else None,
        auto_augment=cfg.augmentation.aa if 'aa' in cfg.augmentation else None,
        num_aug_splits=cfg.augmentation.aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=cfg.data_loader.workers,
        distributed=cfg.distributed,
        collate_fn=collate_fn,
        pin_memory=cfg.data_loader.pin_mem,
        use_multi_epochs_loader=cfg.data_loader.use_multi_epochs_loader,
        use_tencrop=True,
    )

    return loader_train, mixup_active, mixup_fn


def setup_loss(mixup_active):
    if cfg.loss.jsd:
        assert cfg.augmentation.aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=cfg.augmentation.aug_splits, smoothing=cfg.loss.smoothing).cuda()
    elif mixup_active:  # smoothing is handled with mixup label transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif cfg.loss.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=cfg.loss.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()

    return train_loss_fn


def setup_env(args):
    if args.folder is not None:
        cfg.merge_from_file(os.path.join(args.folder, 'eval.yaml'))
    cfg.root_dir = args.folder

    cfg.logger_name = 'eval_log'
    setup_default_logging()

    world_size = 1
    rank = 0  # global rank
    cfg.distributed = torch.cuda.device_count() > 1

    if cfg.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    cfg.num_gpus = world_size

    pop_unused_value(cfg)
    cfg.freeze()

    if cfg.distributed:
        logger_info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (
        rank, cfg.num_gpus))
    else:
        logger_info('Training with a single process on %d GPUs.' % cfg.num_gpus)
    torch.manual_seed(cfg.seed + rank)


def train_epoch(
        epoch, model, bert_model, loader, optimizer, loss_fn, cfg,
        lr_scheduler=None, saver=None, train_meter=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    if cfg.augmentation.mixup_off_epoch and epoch >= cfg.augmentation.mixup_off_epoch:
        if cfg.data_loader.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    model.eval()
    bert_model.train()
    num_updates = epoch * len(loader)
    train_meter.iter_tic()

    for batch_idx, (input, target) in enumerate(loader):
        if not cfg.data_loader.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        with amp_autocast():
            # with torch.no_grad():
                # bs, nc, c, h, w = input.size()
                # output = model(input.view(-1, c, h, w))
                # output = output.softmax(dim=-1)
            output = input.view(-1, cfg.bert.patch_num, input.shape[-1])
            output = bert_model(output)

            loss = loss_fn(output, target)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, parameters=bert_model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if cfg.solver.clip_grad > 0:
                dispatch_clip_grad(
                    model_parameters(bert_model, exclude_head='agc' in cfg.solver.clip_mode),
                    value=cfg.solver.clip_grad, mode=cfg.solver.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(bert_model)

        torch.cuda.synchronize()
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=None)

        num_updates += 1
        loss = dist.scaled_all_reduce([loss.data])[0]
        mb_size = input.size(0) * cfg.num_gpus
        lr_str = str(list(set([param_group['lr'] for param_group in optimizer.param_groups])))
        train_meter.update_stats(loss.item(), lr_str, mb_size)
        train_meter.iter_toc()
        train_meter.log_iter_stats(epoch, batch_idx)
        train_meter.iter_tic()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    train_meter.reset()


def main():
    args = parse_args()
    print('Called with args:')
    setup_env(args)
    if cfg.distributed:
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    model, bert_model, data_config = setup_model()
    optimizer = create_optimizer(cfg, bert_model)

    amp_autocast = suppress  # do nothing

    model, bert_model, model_ema, optimizer, resume_epoch, loss_scaler = setup_resume(args.local_rank, model,
                                                                                      bert_model, optimizer)
    if cfg.data_loader.eval_offline:
        evaler = BertOfflineOpensetEvaler(data_config)
    else:
        evaler = BertOpensetEvaler(data_config)

    logger_info('Vanilla:')
    top1_acc = evaler(resume_epoch, model, bert_model, amp_autocast=amp_autocast)
    #logger_info('EMA:')
    #if model_ema is not None and not cfg.model.model_ema_force_cpu:
    #    if cfg.distributed and cfg.BN.dist_bn in ('broadcast', 'reduce'):
    #        distribute_bn(model_ema, cfg.num_gpus, cfg.BN.dist_bn == 'reduce')
    #    top1_acc = evaler(resume_epoch, model, model_ema.module, amp_autocast=amp_autocast, ema=True)


if __name__ == '__main__':
    main()
