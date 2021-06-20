# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .layers import create_classifier

__all__ = ['SupMoCo']


class SupMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder_q, encoder_k, num_classes=1000, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SupMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.num_classes = num_classes

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # linear cls
        _, self.fc_cls = create_classifier(self.encoder_q.num_features, self.num_classes)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("feat_queue", torch.randn(dim, K))
        self.feat_queue = nn.functional.normalize(self.feat_queue, dim=0)
        self.register_buffer("target_queue", torch.zeros(K) - 1.)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, target):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        targets = concat_all_gather(target)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.feat_queue[:, ptr:ptr + batch_size] = keys.T
        self.target_queue[ptr:ptr + batch_size] = targets.reshape(-1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, target=None, for_test=False):
        if for_test:
            return self.forward_test(im_q)
        return self.forward_train(im_q, im_k, target)

    def forward_test(self, im_q):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        _, feat = self.encoder_q(im_q, return_feats=True)  # queries: NxC

        cls_logits = self.fc_cls(feat)

        return cls_logits

    def forward_train(self, im_q, im_k, target):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, feat = self.encoder_q(im_q, return_feats=True)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        if self.encoder_q.drop_rate:
            feat = F.dropout(feat, p=float(self.encoder_q.drop_rate), training=self.encoder_q.training)
        cls_logits = self.fc_cls(feat)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        k_queue = torch.cat([k.t(), self.feat_queue.clone().detach()], dim=1)
        target_t = torch.cat([target, self.target_queue], dim=0)[None, :]
        labels = torch.eq(target[:, None], target_t).float().to(q.device)  # (N, N+K)

        logits = torch.einsum('nc,ck->nk', [q, k_queue]) / self.T  # (N, N+K)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, target)

        return cls_logits, logits, labels

def _create_sup_moco(**kwargs):
    return SupMoCo(**kwargs)

@register_model
def sup_moco(**kwargs):
    return _create_sup_moco(**kwargs)

# utils
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
