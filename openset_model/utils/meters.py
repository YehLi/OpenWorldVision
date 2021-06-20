import decimal
import numpy as np
from collections import deque
import sklearn.metrics

import torch
from config import cfg
from utils.timer import Timer
from utils.logger import logger_info
import utils.distributed as dist
from utils.distributed import sum_tensor

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 1.0 for k in topk]


def accuracy_trunc(output, target, topk=(1,), thresh=0.1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    _output = output.softmax(dim=1)
    score, pred = _output.topk(maxk, 1, True, True)
    pred = pred.t() + 1
    unknown_inds = (score[:, 0] <= thresh)
    pred[0, unknown_inds] = 0
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 1.0 for k in topk]

def accuracy_trunc_v3(output, target, topk=(1,), thresh=0.1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    _output = output.softmax(dim=1)
    score, pred = _output.topk(maxk, 1, True, True)
    pred = pred.t()
    unknown_inds = (score[:, 0] <= thresh) | (pred[0, :] == 0)
    pred[0, unknown_inds] = 0
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 1.0 for k in topk]


def auroc(known_prob, unknown_prob, in_low=True):
    known_prob, _ = torch.max(known_prob, dim=1)
    unknown_prob, _ = torch.max(unknown_prob, dim=1)

    all_prob = torch.cat([known_prob, unknown_prob], dim=0)
    target = torch.cat([
        known_prob.new_ones(known_prob.size(0)),
        unknown_prob.new_zeros(unknown_prob.size(0))], dim=0)
    assert all_prob.shape == target.shape
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        target.cpu().numpy(), all_prob.cpu().numpy(),
        pos_label=in_low)
    auc = sklearn.metrics.auc(fpr, tpr)

    return torch.tensor(auc, device=all_prob.device)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)

def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024

def float_to_decimal(data, prec=4):
    """Convert floats to decimals which allows for fixed width json."""
    if isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data

class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count

class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, lr, mb_size):
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
    
    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, Iter: {:s}, loss: {:.4f}, lr: {:s}, time_avg: {:.4f}, eta: {:s}, mem: {:d}".format(\
            stats["epoch"], stats["iter"], stats["loss"], stats["lr"], stats["time_avg"], stats["eta"], stats["mem"])
        logger_info(info)


class TrainMeterV2(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_c = ScalarMeter(cfg.solver.log_interval)
        self.loss_o = ScalarMeter(cfg.solver.log_interval)
        self.loss_unknown = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.loss_c_total = 0.0
        self.loss_o_total = 0.0
        self.loss_unknown_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_c.reset()
        self.loss_o.reset()
        self.loss_unknown.reset()
        self.loss_total = 0.0
        self.loss_c_total = 0.0
        self.loss_o_total = 0.0
        self.loss_unknown_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, loss_c, loss_o, loss_unknown, lr, mb_size):
        self.loss.add_value(loss)
        self.loss_c.add_value(loss_c)
        self.loss_o.add_value(loss_o)
        self.loss_unknown.add_value(loss_unknown)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.loss_c_total += loss_c * mb_size
        self.loss_o_total += loss_o * mb_size
        self.loss_unknown_total += loss_unknown * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "loss_c": self.loss_c.get_win_avg(),
            "loss_o": self.loss_o.get_win_avg(),
            "loss_unknown": self.loss_unknown.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, "\
               "Iter: {:s}, "\
               "loss: {:.4f}, "\
               "loss_c: {:.4f}, "\
               "loss_o: {:.4f}, "\
               "loss_unknown: {:.4f}, "\
               "lr: {:s}, "\
               "time_avg: {:.4f}, "\
               "eta: {:s}, "\
               "mem: {:d}".format(
                    stats["epoch"],
                    stats["iter"],
                    stats["loss"],
                    stats["loss_c"],
                    stats["loss_o"],
                    stats["loss_unknown"],
                    stats["lr"],
                    stats["time_avg"],
                    stats["eta"],
                    stats["mem"])
        logger_info(info)


class TrainMeterCNT(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_ce = ScalarMeter(cfg.solver.log_interval)
        self.loss_cnt = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.loss_ce_total = 0.0
        self.loss_cnt_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_ce.reset()
        self.loss_cnt.reset()
        self.loss_total = 0.0
        self.loss_ce_total = 0.0
        self.loss_cnt_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, loss_ce, loss_cnt, lr, mb_size):
        self.loss.add_value(loss)
        self.loss_ce.add_value(loss_ce)
        self.loss_cnt.add_value(loss_cnt)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.loss_ce_total += loss_ce * mb_size
        self.loss_cnt_total += loss_cnt * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "loss_ce": self.loss_ce.get_win_avg(),
            "loss_cnt": self.loss_cnt.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, "\
               "Iter: {:s}, "\
               "loss: {:.4f}, "\
               "loss_ce: {:.4f}, "\
               "loss_cnt: {:.4f}, "\
               "lr: {:s}, "\
               "time_avg: {:.4f}, "\
               "eta: {:s}, "\
               "mem: {:d}".format(
                    stats["epoch"],
                    stats["iter"],
                    stats["loss"],
                    stats["loss_ce"],
                    stats["loss_cnt"],
                    stats["lr"],
                    stats["time_avg"],
                    stats["eta"],
                    stats["mem"])
        logger_info(info)


class TrainMeterSelfSup(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_ce = ScalarMeter(cfg.solver.log_interval)
        self.loss_selfsup = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.loss_ce_total = 0.0
        self.loss_selfsup_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_ce.reset()
        self.loss_selfsup.reset()
        self.loss_total = 0.0
        self.loss_ce_total = 0.0
        self.loss_selfsup_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, loss_ce, loss_selfsup, lr, mb_size):
        self.loss.add_value(loss)
        self.loss_ce.add_value(loss_ce)
        self.loss_selfsup.add_value(loss_selfsup)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.loss_ce_total += loss_ce * mb_size
        self.loss_selfsup_total += loss_selfsup * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "loss_ce": self.loss_ce.get_win_avg(),
            "loss_selfsup": self.loss_selfsup.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, "\
               "Iter: {:s}, "\
               "loss: {:.4f}, "\
               "loss_ce: {:.4f}, "\
               "loss_selfsup: {:.4f}, "\
               "lr: {:s}, "\
               "time_avg: {:.4f}, "\
               "eta: {:s}, "\
               "mem: {:d}".format(
                    stats["epoch"],
                    stats["iter"],
                    stats["loss"],
                    stats["loss_ce"],
                    stats["loss_selfsup"],
                    stats["lr"],
                    stats["time_avg"],
                    stats["eta"],
                    stats["mem"])
        logger_info(info)


class TrainMeterAdv(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_base = ScalarMeter(cfg.solver.log_interval)
        self.loss_adv = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.loss_base_total = 0.0
        self.loss_adv_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_base.reset()
        self.loss_adv.reset()
        self.loss_total = 0.0
        self.loss_base_total = 0.0
        self.loss_adv_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, loss_base, loss_adv, lr, mb_size):
        self.loss.add_value(loss)
        self.loss_base.add_value(loss_base)
        self.loss_adv.add_value(loss_adv)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.loss_base_total += loss_base * mb_size
        self.loss_adv_total += loss_adv * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "loss_base": self.loss_base.get_win_avg(),
            "loss_adv": self.loss_adv.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, "\
               "Iter: {:s}, "\
               "loss: {:.4f}, "\
               "loss_base: {:.4f}, "\
               "loss_adv: {:.4f}, "\
               "lr: {:s}, "\
               "time_avg: {:.4f}, "\
               "eta: {:s}, "\
               "mem: {:d}".format(
                    stats["epoch"],
                    stats["iter"],
                    stats["loss"],
                    stats["loss_base"],
                    stats["loss_adv"],
                    stats["lr"],
                    stats["time_avg"],
                    stats["eta"],
                    stats["mem"])
        logger_info(info)


class TrainMeterPseudo(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_ce = ScalarMeter(cfg.solver.log_interval)
        self.loss_gce = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.loss_ce_total = 0.0
        self.loss_gce_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_ce.reset()
        self.loss_gce.reset()
        self.loss_total = 0.0
        self.loss_ce_total = 0.0
        self.loss_gce_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, loss_ce, loss_gce, lr, mb_size):
        self.loss.add_value(loss)
        self.loss_ce.add_value(loss_ce)
        self.loss_gce.add_value(loss_gce)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.loss_ce_total += loss_ce * mb_size
        self.loss_gce_total += loss_gce * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "loss_ce": self.loss_ce.get_win_avg(),
            "loss_gce": self.loss_gce.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, "\
               "Iter: {:s}, "\
               "loss: {:.4f}, "\
               "loss_ce: {:.4f}, "\
               "loss_gce: {:.4f}, "\
               "lr: {:s}, "\
               "time_avg: {:.4f}, "\
               "eta: {:s}, "\
               "mem: {:d}".format(
                    stats["epoch"],
                    stats["iter"],
                    stats["loss"],
                    stats["loss_ce"],
                    stats["loss_gce"],
                    stats["lr"],
                    stats["time_avg"],
                    stats["eta"],
                    stats["mem"])
        logger_info(info)


class TestMeter(object):
    def __init__(self):
        self.num_top1 = 0
        self.num_top5 = 0
        self.num_samples = 0

    def reset(self):
        self.num_top1 = 0
        self.num_top5 = 0
        self.num_samples = 0

    def update_stats(self, num_top1, num_top5, mb_size):
        self.num_top1 += num_top1
        self.num_top5 += num_top5
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch):
        if cfg.distributed:
            tensor_reduce = torch.tensor([self.num_top1 * 1.0, self.num_top5 * 1.0, self.num_samples * 1.0], device="cuda")
            tensor_reduce = sum_tensor(tensor_reduce)
            tensor_reduce = tensor_reduce.data.cpu().numpy()
            num_top1 = tensor_reduce[0]
            num_top5 = tensor_reduce[1]
            num_samples = tensor_reduce[2]
        else:
            num_top1 = self.num_top1
            num_top5 = self.num_top5
            num_samples = self.num_samples

        top1_acc = num_top1 * 1.0 / num_samples
        top5_acc = num_top5 * 1.0 / num_samples

        info = "Epoch: {:d}, top1_acc = {:.2%}, top5_acc = {:.2%} in {:d}".format(cur_epoch + 1, top1_acc, top5_acc, int(num_samples))
        logger_info(info)
        return top1_acc, top5_acc


class TestAvgClsMeter(object):
    def __init__(self, num_cls):
        self.num_top1 = torch.zeros(num_cls, device="cuda")
        self.num_samples = torch.zeros(num_cls, device="cuda")
        self.num_cls = num_cls

    def reset(self):
        self.num_top1 = torch.zeros(self.num_cls, device="cuda")
        self.num_samples = torch.zeros(self.num_cls, device="cuda")

    def update_stats(self, num_top1, mb_size):
        self.num_top1 += num_top1
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch):
        if cfg.distributed:
            tensor_reduce = torch.stack([self.num_top1 * 1.0, self.num_samples * 1.0], dim=0)
            tensor_reduce = sum_tensor(tensor_reduce)
            tensor_reduce = tensor_reduce.data.cpu().numpy()
            num_top1 = tensor_reduce[0]
            num_samples = tensor_reduce[1]
        else:
            num_top1 = self.num_top1
            num_samples = self.num_samples

        top1_acc = (num_top1 * 1.0 / num_samples).mean()
        unknown_acc = num_top1[0] * 1.0 / num_samples[0]

        info = "Epoch: {:d}, top1_acc = {:.2%} in {:d} unknown_acc {:.2%} in {:d}".format(
            cur_epoch + 1, top1_acc, int(num_samples.sum()), unknown_acc, int(num_samples[0]))
        logger_info(info)
        return top1_acc


class TestAvgClsAUROCMeter(object):
    def __init__(self, num_cls):
        self.num_top1 = torch.zeros(num_cls, device="cuda")
        self.num_samples = torch.zeros(num_cls, device="cuda")
        self.num_cls = num_cls

    def reset(self):
        self.num_top1 = torch.zeros(self.num_cls, device="cuda")
        self.num_samples = torch.zeros(self.num_cls, device="cuda")

    def update_stats(self, num_top1, mb_size):
        self.num_top1 += num_top1
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, auroc):
        if cfg.distributed:
            tensor_reduce = torch.stack([self.num_top1 * 1.0, self.num_samples * 1.0], dim=0)
            tensor_reduce = sum_tensor(tensor_reduce)
            tensor_reduce = tensor_reduce.data.cpu().numpy()
            num_top1 = tensor_reduce[0]
            num_samples = tensor_reduce[1]
        else:
            num_top1 = self.num_top1
            num_samples = self.num_samples

        top1_acc = (num_top1 * 1.0 / num_samples).mean()
        unknown_acc = num_top1[0] * 1.0 / num_samples[0]
        known_acc = (num_top1[1:] * 1.0 / num_samples[1:]).mean()

        info = "Epoch: {:d}, top1_acc = {:.2%} in {:d} known_acc {:.2%} in {:d}, unknown_acc {:.2%} in {:d}, auroc {:.2%}.".format(
            cur_epoch + 1, top1_acc, int(num_samples.sum()),
            known_acc, int(num_samples[1:].sum()),
            unknown_acc, int(num_samples[0]),
            auroc.cpu().numpy())
        logger_info(info)
        return top1_acc


class OpenSetTestMeter(object):
    def __init__(self, num_cls):
        self.num_top1 = torch.zeros(num_cls, device="cuda")
        self.num_samples = torch.zeros(num_cls, device="cuda")
        self.num_cls = num_cls

    def reset(self):
        self.num_top1 = torch.zeros(self.num_cls, device="cuda")
        self.num_samples = torch.zeros(self.num_cls, device="cuda")

    def update_stats(self, num_top1, mb_size):
        self.num_top1 += num_top1
        self.num_samples += mb_size

    def log_iter_stats(self,
                       cur_epoch,
                       top1,
                       auroc,
                       auprc,
                       f1,
                       ccr):

        info = "Epoch: {:d}, macro_top1 = {:.2%}, "\
               "micro_top1 {:.2%}, "\
               "unknown_top1 {:.2%}, "\
               "auroc {:.2%}, "\
               "auprc {:.2%}, "\
               "f1 {:.2%}, "\
               "ccr {:.2%}, ".format(cur_epoch + 1,
                                     top1['macro_top1'],
                                     top1['micro_top1'],
                                     top1['unknown_top1'],
                                     auroc,
                                     auprc,
                                     f1,
                                     ccr)
        logger_info(info)
        return top1['macro_top1']