import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, AvgPool2dSame, ConvBnAct, SEModule, DropPath, BlurPool2d
from .registry import register_model
from cupy_layers.aggregation_zeropad import LocalConvolution
from .layers import get_act_layer

def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    regnetx_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    regnetx_004=_mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    regnetx_006=_mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    regnetx_008=_mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    regnetx_016=_mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    regnetx_032=_mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    regnetx_040=_mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    regnetx_064=_mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    regnetx_080=_mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    regnetx_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    regnetx_160=_mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    regnetx_320=_mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    regnety_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    regnety_004=_mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    regnety_006=_mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    regnety_008=_mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    regnety_016=_mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    regnety_032=_mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    regnety_040=_mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    regnety_064=_mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    regnety_080=_mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    regnety_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    regnety_160=_mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    regnety_320=_mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25),
)


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
    }


default_cfgs = dict(
    regnetx_002=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_002-e7e85e5c.pth'),
    regnetx_004=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_004-7d0e9424.pth'),
    regnetx_006=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_006-85ec1baa.pth'),
    regnetx_008=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_008-d8b470eb.pth'),
    regnetx_016=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth'),
    regnetx_032=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_032-ed0c7f7e.pth'),
    regnetx_040=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_040-73c2a654.pth'),
    regnetx_064=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_064-29278baa.pth'),
    regnetx_080=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_080-7c7fcab1.pth'),
    regnetx_120=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_120-65d5521e.pth'),
    regnetx_160=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_160-c98c4112.pth'),
    regnetx_320=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_320-8ea38b93.pth'),
    regnety_002=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_002-e68ca334.pth'),
    regnety_004=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_004-0db870e6.pth'),
    regnety_006=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_006-c67e57ec.pth'),
    regnety_008=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_008-dc900dbe.pth'),
    regnety_016=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth'),
    regnety_032=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/regnety_032_ra-7f2439f9.pth'),
    regnety_040=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_040-f0d569f9.pth'),
    regnety_064=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_064-0a48325c.pth'),
    regnety_080=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_080-e7f3eb93.pth'),
    regnety_120=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_120-721ba79a.pth'),
    regnety_160=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_160-d64013cd.pth'),
    regnety_320=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_320-ba464b29.pth'),
)


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
    return widths, num_stages, max_stage, widths_cont

class TransLayer(nn.Module):
    def __init__(self, dim, kernel_size, groups):
        super(TransLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=groups, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        
        return out.contiguous()

class CoTBottleneck(nn.Module):
    """ RegNet Bottleneck
    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, bottleneck_ratio=1, group_width=1, se_ratio=0.25,
                 downsample=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(CoTBottleneck, self).__init__()
        bottleneck_chs = int(round(out_chs * bottleneck_ratio))
        groups = bottleneck_chs // group_width

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        #self.conv2 = ConvBnAct(
        #    bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation,
        #    groups=groups, **cargs)
        
        self.conv2 = TransLayer(bottleneck_chs, kernel_size=3, groups=groups)
        if stride > 1:
            self.avd = nn.AvgPool2d(3, stride, padding=1) if aa_layer is None else aa_layer(channels=bottleneck_chs, stride=stride)
        else:
            self.avd = None

        cargs['act_layer'] = None
        self.conv3 = ConvBnAct(bottleneck_chs, out_chs, kernel_size=1, **cargs)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


class Bottleneck(nn.Module):
    """ RegNet Bottleneck
    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, bottleneck_ratio=1, group_width=1, se_ratio=0.25,
                 downsample=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        bottleneck_chs = int(round(out_chs * bottleneck_ratio))
        groups = bottleneck_chs // group_width

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvBnAct(
            bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation,
            groups=groups, **cargs)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, reduction_channels=se_channels)
        else:
            self.se = None
        cargs['act_layer'] = None
        self.conv3 = ConvBnAct(bottleneck_chs, out_chs, kernel_size=1, **cargs)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


def downsample_conv(
        in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    return ConvBnAct(
        in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, norm_layer=norm_layer, act_layer=None)


def downsample_avg(
        in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(*[
        pool, ConvBnAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, act_layer=None)])


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio, group_width,
                 block_fn=Bottleneck, se_ratio=0., drop_path_rates=None, drop_block=None, aa_layer=None, stage_id=0):
        super(RegStage, self).__init__()
        block_kwargs = {}  # FIXME setup to pass various aa, norm, act layer common args
        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = first_dilation if i == 0 else dilation
            if drop_path_rates is not None and drop_path_rates[i] > 0.:
                drop_path = DropPath(drop_path_rates[i])
            else:
                drop_path = None
            if (block_in_chs != out_chs) or (block_stride != 1):
                #proj_block = downsample_conv(block_in_chs, out_chs, 1, block_stride, block_dilation)
                proj_block = downsample_avg(block_in_chs, out_chs, 1, block_stride, block_dilation)
            else:
                proj_block = None

            name = "b{}".format(i + 1)

            if stage_id == 2 and (i == depth//2 or i == 0):
                self.add_module(
                    name, CoTBottleneck(
                        block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio,
                        downsample=proj_block, drop_block=drop_block, drop_path=drop_path, **block_kwargs)
                )
            elif stage_id == 3 and i % 2 == 0:
                self.add_module(
                    name, CoTBottleneck(
                        block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio,
                        downsample=proj_block, drop_block=drop_block, drop_path=drop_path, **block_kwargs)
                )
            else:
                self.add_module(
                    name, block_fn(
                        block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio,
                        downsample=proj_block, drop_block=drop_block, drop_path=drop_path, **block_kwargs)
                )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class RegCotNet(nn.Module):
    """RegNet model.
    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.,
                 drop_path_rate=0., aa_layer=None, zero_init_last_bn=True):
        super().__init__()
        # TODO add drop block, drop path, anti-aliasing, custom bn/act args
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)

        # Construct the stem
        stem_width = cfg['stem_width']
        #self.stem = ConvBnAct(in_chans, stem_width, 3, stride=2)
        
        #############################################################################
        # Stem
        stem_chs_1 = stem_chs_2 = stem_width
        self.conv1 = nn.Sequential(*[
            ConvBnAct(in_chans, stem_chs_1, 3, stride=2),
            ConvBnAct(stem_chs_1, stem_chs_2, 3, stride=1),
            ConvBnAct(stem_chs_2, stem_width, 3, stride=1)
        ])
        #############################################################################

        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 2
        stage_params = self._get_stage_params(cfg, output_stride=output_stride, drop_path_rate=drop_path_rate)
        se_ratio = cfg['se_ratio']
        for i, stage_args in enumerate(stage_params):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(prev_width, **stage_args, se_ratio=se_ratio, aa_layer=aa_layer, stage_id=i))
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        self.num_features = prev_width
        self.head = ClassifierHead(
            in_chs=prev_width, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _get_stage_params(self, cfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg['wa'], cfg['w0'], cfg['wm'], cfg['depth']
        widths, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)

        # Use the same group width, bottleneck mult and stride for each stage
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_bottle_ratios = [cfg['bottle_ratio'] for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, d), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_groups = adjust_widths_groups_comp(stage_widths, stage_bottle_ratios, stage_groups)
        param_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_width', 'drop_path_rates']
        stage_params = [
            dict(zip(param_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_bottle_ratios, stage_groups,
                stage_dpr)]
        return stage_params

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


def _create_regcotnet(variant, pretrained, **kwargs):
    return build_model_with_cfg(
        RegCotNet, variant, pretrained, default_cfg=default_cfgs[variant], model_cfg=model_cfgs[variant], **kwargs)

@register_model
def regcotnety_040(pretrained=False, **kwargs):
    """RegNetY-4.0GF"""
    model_args = dict(aa_layer=BlurPool2d, **kwargs)
    return _create_regcotnet('regnety_040', pretrained, **model_args)

@register_model
def regcotnety_120(pretrained=False, **kwargs):
    """RegNetY-12GF"""
    model_args = dict(aa_layer=BlurPool2d, **kwargs)
    return _create_regcotnet('regnety_120', pretrained, **model_args)

@register_model
def regcotnety_160(pretrained=False, **kwargs):
    """RegNetY-16GF"""
    model_args = dict(aa_layer=BlurPool2d, **kwargs)
    return _create_regcotnet('regnety_160', pretrained, **model_args)

@register_model
def regcotnety_320(pretrained=False, **kwargs):
    """RegNetY-32GF"""
    model_args = dict(aa_layer=BlurPool2d, **kwargs)
    return _create_regcotnet('regnety_320', pretrained, **model_args)
