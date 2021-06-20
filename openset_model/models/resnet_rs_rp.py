import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, create_attn, create_classifier
from .registry import register_model
from .resnet_rs import ResNetRS, Bottleneck


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'resnet_rs50_rp': _cfg(
        url='',
        input_size=(3, 224, 224)),
    'resnet_rs101_rp': _cfg(
        url='',
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.857, interpolation='bicubic'),
    'resnet_rs152_rp': _cfg(
        url='',
        input_size=(3, 224, 224)),
}


class ResNetRSRP(ResNetRS):
    def __init__(self,
                 gamma=0.2,
                 num_rp_per_cls=8,
                 *args,
                 **kwargs):
        super(ResNetRSRP, self).__init__(*args, **kwargs)
        self.num_rp_per_cls = num_rp_per_cls
        self.P = nn.Parameter(
            torch.zeros((self.num_classes * self.num_rp_per_cls, self.num_features)))
        self.R = nn.Parameter(
            torch.zeros((self.num_classes,)))
        self.fc = None
        self.gamma = gamma

        nn.init.normal_(self.P)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        # compute dist to reciprocal points
        dist_to_rp = torch.cdist(x, self.P, p=2).square().reshape(
            x.size(0), self.num_classes, self.num_rp_per_cls)  # (N, C, 8)
        logits = dist_to_rp.mean(dim=2) * self.gamma
        # dist_to_rp = nn.functional.normalize(dist_to_rp, dim=1)
        # R = nn.functional.normalize(self.R, dim=0)
        # open dist
        x_norm = nn.functional.normalize(x, dim=1)
        P_norm = nn.functional.normalize(self.P, dim=1)
        dist_to_rp_norm = torch.cdist(x_norm, P_norm, p=2).square().reshape(
            x.size(0), self.num_classes, self.num_rp_per_cls)  # (N, C, 8)
        open_dist = (dist_to_rp_norm - self.R[None, :, None]).square()
        return logits, dist_to_rp_norm, open_dist


def _create_resnet_rs_rp(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNetRSRP, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)


@register_model
def resnet_rs50_rp(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  
    stem_type='deep', stem_width=32, base_width=64, cardinality=1,
    block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet_rs_rp('resnet_rs50_rp', pretrained, **model_args)


@register_model
def resnet_rs101_rp(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3],  
    stem_type='deep', stem_width=64, base_width=64, cardinality=1,
    block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet_rs_rp('resnet_rs101_rp', pretrained, **model_args)


@register_model
def resnet_rs101L_rp(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3],  
    stem_type='deep', stem_width=64, base_width=64, cardinality=1,
    block_args=dict(attn_layer='se'), **kwargs)
    return _create_resnet_rs_rp('resnet_rs152_rp', pretrained, **model_args)