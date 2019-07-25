import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF

__all__ = ['SVX_CNN',]

def init_conv3d(num_in, num_out, kernel_size=3, padding=1, stride=1):
    conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding, stride=stride)
    conv.weight.data.normal_(std=0.001)
    conv.bias.data.fill_(0)
    return conv

class Conv3d_bn_relu(nn.Module):
    def __init__(self, num_in, num_out, use_bn=True, use_affine=False):
        super(Conv3d_bn_relu, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.use_bn = use_bn
        self.use_affine = use_affine
        # conv1
        self.conv1 = conv1 = init_conv3d(num_in, num_out, kernel_size=3, padding=1, stride=1)
        # bn1
        if use_bn:
            self.bn1 = bn1 = nn.BatchNorm3d(num_out, eps=1e-5, affine=use_affine)
            if use_affine:
                bn1.weight.data.fill_(1)
                bn1.bias.data.fill_(0)

    def forward(self, x):
        if self.use_bn:
            return FF.relu(self.bn1(self.conv1(x)), inplace=True)
        else:
            return FF.relu(self.conv1(x), inplace=True)

def conv3d_bn_relu_layer(num_in, num_out):
    return Conv3d_bn_relu(num_in, num_out, use_bn=True, use_affine=False)

def conv3d_relu_layer(num_in, num_out):
    return Conv3d_bn_relu(num_in, num_out, use_bn=False)

def caffeZoom3d(x, zoom_factor, mode='trilinear'):
    B, C, L, H, W = x.shape
    LL = L + (L-1)*(zoom_factor-1)
    HH = H + (H-1)*(zoom_factor-1)
    WW = W + (W-1)*(zoom_factor-1)
    return FF.interpolate(x, size=[LL, HH, WW], mode=mode, align_corners=True)

def caffeCrop3d_as(x, ref_shape):
    _, _, Lx, Hx, Wx = x.shape
    _, _, L, H, W = ref_shape
    assert Lx >= L and Hx >= H and Wx >= W, "size of x should not be smaller than ref"
    return x[:, :, :L, :H, :W]

class SVX_CNN(nn.Module):
    def __init__(self, num_in, num_out, num_ch=32):
        super(SVX_CNN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.num_ch = num_ch
        # 
        self.conv1 = conv3d_bn_relu_layer(num_in, num_ch)
        self.conv2 = conv3d_bn_relu_layer(num_ch, num_ch)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        #
        self.conv3 = conv3d_bn_relu_layer(num_ch, num_ch)
        self.conv4 = conv3d_bn_relu_layer(num_ch, num_ch)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        #
        self.conv5 = conv3d_bn_relu_layer(num_ch, num_ch)
        self.conv6 = conv3d_bn_relu_layer(num_ch, num_ch)
        #
        self.conv7_x  = init_conv3d(num_in, num_out)
        self.conv7_c2 = init_conv3d(num_ch, num_out)
        self.conv7_c4 = init_conv3d(num_ch, num_out)
        self.conv7_c6 = init_conv3d(num_ch, num_out)

    def forward(self, x):
        y = self.conv7_x(x)
        x = self.conv2(self.conv1(x))  # conv2
        _shape = x.shape
        y += self.conv7_c2(x)
        x = self.conv4(self.conv3(self.pool1(x)))  # conv4
        y += self.conv7_c4(caffeCrop3d_as(caffeZoom3d(x, zoom_factor=2), _shape))
        y += self.conv7_c6(caffeCrop3d_as(caffeZoom3d(self.conv6(self.conv5(self.pool2(x))), zoom_factor=4), _shape))
        return FF.relu(y, inplace=True)