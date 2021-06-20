import os
import numpy as np
from config import cfg
from datasets import DatasetTxt, create_loader_index
from utils.meters import OpenSetTestMeter
import torch

from .openset_eval import get_accuracy, roc, ccr, macro_F1, pr


class TenCropOpensetEvaler(object):
    def __init__(self, data_config):
        super(TenCropOpensetEvaler, self).__init__()
        self.loaders_eval = self.build_dataset(data_config)
        self.num_classes = cfg.model.num_classes

    def build_dataset(self, data_config):
        assert os.path.isdir(cfg.data_loader.data_path)
        if not isinstance(cfg.data_loader.eval_label, list):
            cfg.data_loader.eval_label = [cfg.data_loader.eval_label]
        loaders_eval = []
        for eval_label in cfg.data_loader.eval_label:
            dataset_eval = DatasetTxt(cfg.data_loader.data_path, eval_label, return_path=True)

            loader_eval = create_loader_index(
                dataset_eval,
                input_size=data_config['input_size'],
                batch_size=cfg.data_loader.vbatch_size,
                is_training=False,
                use_prefetcher=False,
                interpolation=data_config['interpolation'],
                mean=data_config['mean'],
                std=data_config['std'],
                num_workers=cfg.data_loader.workers,
                distributed=cfg.distributed,
                crop_pct=data_config['crop_pct'],
                pin_memory=cfg.data_loader.pin_mem,
                use_tencrop=True,
            )
            loaders_eval.append(loader_eval)
        return loaders_eval

    def save_results(self, epoch, loader_eval, eval_idx, ema, cls_preds_cat, indexs_cat):
        imageList = loader_eval.dataset.img_names
        basename = cfg.data_loader.eval_label[eval_idx].split('/')[-1][:-4]
        if ema:
            save_name = 'ema_tencrop_pred_' + basename + '-ep{}.csv'.format(epoch)
        else:
            save_name = 'vanilla_tencrop_pred_' + basename + '-ep{}.csv'.format(epoch)

        save_dir = os.path.join(cfg.root_dir, 'results')
        if cfg.distributed:
            if torch.distributed.get_rank() == 0:
                os.makedirs(save_dir, exist_ok=True)
                file_pred = os.path.join(save_dir, save_name)
                result_str = ''
                for i, im_path in enumerate(imageList):
                    result_str += im_path.split('/')[-1]
                    prob = cls_preds_cat[indexs_cat == i].data.cpu().numpy().ravel()
                    for j in range(self.num_classes):
                        result_str += ',{:.6f}'.format(prob[j])
                    result_str += '\n'
                with open(file_pred, 'w') as f:
                    f.write(result_str)
                    f.close()
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_pred = os.path.join(save_dir, save_name)
            result_str = ''
            for i, im_path in enumerate(imageList):
                result_str += im_path.split('/')[-1]
                prob = cls_preds_cat[indexs_cat == i].data.cpu().numpy().ravel()
                for j in range(self.num_classes):
                    result_str += ',{:.3f}'.format(prob[j])
                result_str += '\n'
            with open(file_pred, 'w') as f:
                f.write(result_str)
                f.close()

    def __call__(self, epoch, model, amp_autocast, ema=False):
        test_meter = OpenSetTestMeter(self.num_classes)
        model.eval()

        top1_acc_list = []
        img_ids = []
        preds = []
        for eval_idx, loader_eval in enumerate(self.loaders_eval):
            with torch.no_grad():
                test_meter.reset()
                cls_preds = []
                targets = []
                indexs = []
                for batch_idx, (input, target, index) in enumerate(loader_eval):
                    bs, nc, c, h, w = input.size()
                    input = input.cuda()
                    target = target.cuda()
                    index = index.cuda()

                    with amp_autocast():
                        output_tencrop = model(input.view(-1, c, h, w))
                        output_tencrop = output_tencrop.softmax(dim=1)
                        output = output_tencrop.view(bs, nc, -1).mean(dim=1)
                    cls_preds.append(output)
                    targets.append(target)
                    indexs.append(index)

                # concat all predictions and targets
                cls_preds = torch.cat(cls_preds, dim=0)
                targets = torch.cat(targets, dim=0)
                indexs = torch.cat(indexs, dim=0)
                cls_preds_cat = concat_all_gather(cls_preds)
                targets_cat = concat_all_gather(targets)
                indexs_cat = concat_all_gather(indexs)

                preds.append(cls_preds_cat)
                img_ids.append(indexs_cat)

                # tensor to numpy
                valid_inds = (targets_cat < self.num_classes)
                preds_np = cls_preds_cat[valid_inds].cpu().numpy()
                preds_binary_np = preds_np[:, 0]
                targets_np = targets_cat[valid_inds].cpu().numpy()
                targets_binary_np = (targets_np == 0)

                # top1 mean cls
                raw_top1 = get_accuracy(preds_np, targets_np, k=(1,), gt_shift=0)[1]
                macro_top1 = np.array([raw_top1[i] for i in range(414)]).mean()
                top1 = dict(
                    macro_top1=macro_top1,
                    micro_top1=raw_top1['all'],
                    unknown_top1=raw_top1[0])
                # auroc
                auroc = roc(preds_binary_np, targets_binary_np, gt_shift=0)['auc']
                # auprc
                auprc = pr(preds_binary_np, targets_binary_np, gt_shift=0)['auc']
                # macro f1
                f1_score = macro_F1(preds_np, targets_np, gt_shift=0)
                # ccr
                ccr_score = ccr(preds_np, targets_np, gt_shift=0)['auc']

                if cfg.distributed:
                    torch.distributed.barrier()

                top1_acc = test_meter.log_iter_stats(epoch, top1, auroc, auprc, f1_score, ccr_score)
                top1_acc_list.append(top1_acc)

                # save results
                self.save_results(epoch, loader_eval, eval_idx, ema, cls_preds_cat, indexs_cat)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        img_ids = torch.cat(img_ids, dim=0).cpu().numpy()
        imageList = loader_eval.dataset.img_names
        filenames = [imageList[im_id] for im_id in img_ids]

        return filenames, preds


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output