from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_module
from timm.models.helpers import named_apply, build_model_with_cfg
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp
from timm.models.registry import register_model
import timm
from timm.models.layers import create_conv2d, SqueezeExcite, MixedConv2d, DropPath

__all__ = ['NEXcepTion']  # model_registry will add each entrypoint fn to this


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=0, bias=bias)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, dim, strides=1, drop_path=0):
        super(BottleneckBlock, self).__init__()


        self.sepconv1 = SeparableConv2d(dim, dim * 3, kernel_size=5, stride=strides, padding=2)
        # if 1 in normaliztion_pos:
        self.norm1 = nn.BatchNorm2d(dim * 3)


        # if 2 in activation_pos:
        self.act = nn.GELU()

        self.sepconv2 = SeparableConv2d(dim * 3, dim, kernel_size=5, stride=strides, padding=2)
        # if 2 in normaliztion_pos:
        self.norm2 = nn.BatchNorm2d(dim)


        self.sepconv3 = SeparableConv2d(dim, dim, kernel_size=5, stride=strides, padding=2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.block_end = SqueezeExcite(dim, act_layer=timm.models.layers.activations.GELU, norm_layer=nn.BatchNorm2d)


    def forward(self, x):
        skip = x
        x = self.sepconv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.sepconv2(x)
        x = self.norm2(x)
        x = self.sepconv3(x)
        x = self.block_end(x)
        x = self.drop_path(x) + skip
        return x

class NEXcepTionStage(nn.Module):

    def __init__(
            self, in_chs, out_chs, stride=2, depth=2, dp_rates=None, norm_layer=None):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
            )
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.Sequential(*[BottleneckBlock(
            dim=out_chs, drop_path=dp_rates[j])
            for j in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)

        x = self.blocks(x)
        return x


class NEXception(nn.Module):
    def __init__(
            self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, patch_size=4,
            depths=(3, 4, 9, 3), dims=(96, 192, 384, 768),  ls_init_value=1e-6, conv_mlp=False, stem_type='patch',
            head_init_scale=1., head_norm_first=False, norm_layer=None, drop_rate=0., drop_path_rate=0.,
    ):
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

 
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_layer(dims[0])
        )
        curr_stride = patch_size
        prev_chs = dims[0]


        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            out_chs = dims[i]
            stages.append(NEXcepTionStage(
                prev_chs, out_chs, stride=stride,
                depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                norm_layer=norm_layer, cl_norm_layer=cl_norm_layer)
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

        # named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict


def _create_convnext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        NEXcepTion, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


