from numpy import pad
import timm.models.layers
from torch import norm
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import create_conv2d, SqueezeExcite, MixedConv2d, DropPath
import functools


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


class NEXception(nn.Module):
    def __init__(self, num_classes, drop_path_rate=0):
        super(NEXception, self).__init__()


        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=2, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(96)
        )
        self.num_classes = num_classes
        self.downsampling_block1 = Block(96, 128, reps=2,
                                         strides=2, drop_path=drop_path_rate)
        self.downsampling_block2 = Block(128, 256, 2, strides=2, drop_path=drop_path_rate)

        self.downsampling_block3 = Block(256, 512, 2, strides=2, drop_path=drop_path_rate)

        self.middle_flow = nn.Sequential(
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
            BottleneckBlock(strides=1, drop_path=drop_path_rate),
        )

        self.exit_flow = nn.Sequential(
            Block(512, 1024, 2, 2, grow_first=False,
                  drop_path=drop_path_rate),
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.GELU(),
            SeparableConv2d(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.GELU()
        )

        self.global_pool, self.fc = timm.models.layers.create_classifier(2048, self.num_classes,
                                                                         pool_type='avg')

    def forward(self, x):
        x = self.stem(x)
        x = self.downsampling_block1(x)
        x = self.downsampling_block2(x)
        x = self.downsampling_block3(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x



# # Modified block from Xception middle flow
class BottleneckBlock(nn.Module):
    def __init__(self, strides=1, drop_path=0):
        super(BottleneckBlock, self).__init__()


        self.sepconv1 = SeparableConv2d(512, 1536, kernel_size=5, stride=strides, padding=2)

        self.norm1 = nn.BatchNorm2d(1536)

        self.act = nn.GELU()

        self.sepconv2 = SeparableConv2d(1536, 512, kernel_size=5, stride=strides, padding=2)

        self.norm2 = nn.BatchNorm2d(512)

        self.sepconv3 = SeparableConv2d(512, 512, kernel_size=5, stride=strides, padding=2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.block_end = SqueezeExcite(512, act_layer=timm.models.layers.activations.GELU, norm_layer=nn.BatchNorm2d)


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


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1,
                 grow_first=True, drop_path=0):
        super(Block, self).__init__()

        normaliztion_pos = [1, 2]
        activation_pos = [2]

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False, padding=0)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            if i + 1 in activation_pos:
                rep.append(nn.GELU())

            rep.append(SeparableConv2d(inc, outc, kernel_size=5, stride=1, padding=2))
            if i + 1 in normaliztion_pos:
                rep.append(nn.BatchNorm2d(outc))

        if strides != 1:
            rep.append(nn.MaxPool2d((3, 3), stride=1, padding=1))
            rep.append(timm.models.layers.BlurPool2d(out_channels, filt_size=3, stride=2))

        self.rep = nn.Sequential(*rep)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.block_end = SqueezeExcite(out_channels, act_layer=timm.models.layers.activations.GELU, norm_layer=nn.BatchNorm2d)

    def forward(self, inp):
        x = self.rep(inp)
        x = self.block_end(x)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.drop_path(x) + skip
        return x


