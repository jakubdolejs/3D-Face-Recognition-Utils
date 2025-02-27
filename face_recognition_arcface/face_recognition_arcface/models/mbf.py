# -*- coding: utf-8 -*-
# @Author        : Cuong Tran
# @Time          : 10/19/2022


import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
        self.relu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(Linear, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)

    def forward(self, x):
        return self.conv2d(x)


class ConvOnly(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(ConvOnly, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)

    def forward(self, x):
        return self.conv2d(x)


class DResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1):
        super(DResidual, self).__init__()
        self.conv_sep = Conv(in_channels, groups, kernel_size=1, padding=0, stride=1)
        self.conv_dw = Conv(groups, groups, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.conv_proj = Linear(groups, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv_sep(x)
        x = self.conv_dw(x)
        x = self.conv_proj(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channels, num_block, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(Residual, self).__init__()
        self.block = nn.ModuleList([
            DResidual(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride, padding, groups)
            for i in range(num_block)
        ])

    def forward(self, x):
        identity = x
        for block in self.block:
            x = block(identity)
            x += identity
            identity = x
        return x


class Network(nn.Module):
    def __init__(self, num_classes=128):
        super(Network, self).__init__()
        self.conv_1 = Conv(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv_2_dw = Conv(64, 64, kernel_size=3, stride=1, padding=1, groups=64)

        self.dconv_23 = DResidual(64, 64, kernel_size=3, stride=2, padding=1, groups=128)
        self.res_3 = Residual(64, 4, 64, kernel_size=3, stride=1, padding=1, groups=128)

        self.dconv_34 = DResidual(64, 128, kernel_size=3, stride=2, padding=1, groups=256)
        self.res_4 = Residual(128, 6, 128, kernel_size=3, stride=1, padding=1, groups=256)

        self.dconv_45 = DResidual(128, 128, kernel_size=3, stride=2, padding=1, groups=512)
        self.res_5 = Residual(128, 2, 128, kernel_size=3, stride=1, padding=1, groups=256)

        self.conv_6sep = Conv(128, 512, kernel_size=1, stride=1, padding=0)
        self.conv_6dw7_7 = ConvOnly(512, 512, kernel_size=7, stride=1, padding=0, groups=512)

        self.pre_fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = (x - 127.5) * 0.007812
        x = self.conv_1(x)
        x = self.conv_2_dw(x)

        x = self.dconv_23(x)
        x = self.res_3(x)

        x = self.dconv_34(x)
        x = self.res_4(x)

        x = self.dconv_45(x)
        x = self.res_5(x)

        x = self.conv_6sep(x)
        x = self.conv_6dw7_7(x)
        x = x.view(x.size(0), -1)
        x = self.pre_fc1(x)
        return x


def create_model():
    model = Network()
    return model
