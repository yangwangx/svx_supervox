import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF

__all__ = ['SSN_CNN',]

class Conv_bn_relu(nn.Module):
    def __init__(self, num_in, num_out, use_bn=True, use_affine=False):
        super(Conv_bn_relu, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.use_bn = use_bn
        self.use_affine = use_affine
        # conv1
        self.conv1 = conv1 = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, stride=1)
        conv1.weight.data.normal_(std=0.001)
        conv1.bias.data.fill_(0)
        # bn1
        if use_bn:
            self.bn1 = bn1 = nn.BatchNorm2d(num_out, eps=1e-5, affine=use_affine)
            if use_affine:
                bn1.weight.data.fill_(1)
                bn1.bias.data.fill_(0)

    def forward(self, x):
        if self.use_bn:
            return FF.relu(self.bn1(self.conv1(x)), inplace=True)
        else:
            return FF.relu(self.conv1(x), inplace=True)

def conv_bn_relu_layer(num_in, num_out):
    return Conv_bn_relu(num_in, num_out, use_bn=True, use_affine=False)

def conv_relu_layer(num_in, num_out):
    return Conv_bn_relu(num_in, num_out, use_bn=False)

def caffeZoom2d(x, zoom_factor, mode='bilinear'):
    B, C, H, W = x.shape
    HH = H + (H-1)*(zoom_factor-1)
    WW = W + (W-1)*(zoom_factor-1)
    return FF.interpolate(x, size=[HH, WW], mode=mode, align_corners=True)

def caffeCrop2d_as(x, ref_shape):
    _, _, Hx, Wx = x.shape
    _, _, H, W = ref_shape
    assert Hx >= H and Wx >= W, "size of x should not be smaller than ref"
    return x[:, :, :H, :W]

class SSN_CNN(nn.Module):
    def __init__(self, num_in, num_out, num_ch=64, pretrained_npz=''):
        super(SSN_CNN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.num_ch = num_ch
        # 
        self.conv1 = conv_bn_relu_layer(num_in, num_ch)
        self.conv2 = conv_bn_relu_layer(num_ch, num_ch)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        #
        self.conv3 = conv_bn_relu_layer(num_ch, num_ch)
        self.conv4 = conv_bn_relu_layer(num_ch, num_ch)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        #
        self.conv5 = conv_bn_relu_layer(num_ch, num_ch)
        self.conv6 = conv_bn_relu_layer(num_ch, num_ch)
        #
        self.conv7 = conv_relu_layer(num_in + num_ch*3, num_out)
        if pretrained_npz != '':
            self.load_pretrained_npz(pretrained_npz=pretrained_npz)
    
    def load_pretrained_npz(self, pretrained_npz):
        print('loading pretrained CNN module from [{}]'.format(pretrained_npz))
        # load pretrained weights
        L = np.load(pretrained_npz, encoding='latin1')
        convs, bns = L['arr_0'], L['arr_1']
        # put pretrained weights at right place
        D = self.state_dict()
        conv_keys = list(k for k in D if 'conv1.weight' in k or 'conv1.bias' in k)
        bn_keys = list(k for k in D if 'bn1.running' in k)
        for k, w in zip(conv_keys, convs):
            D[k].copy_(torch.from_numpy(w))
        for k, w in zip(bn_keys, bns):
            D[k].copy_(torch.from_numpy(w))
        self.load_state_dict(D, strict=False)

    def forward(self, x):
        conv2 = self.conv2(self.conv1(x))
        conv4 = self.conv4(self.conv3(self.pool1(conv2)))
        conv6 = self.conv6(self.conv5(self.pool2(conv4)))
        conv4 = caffeCrop2d_as(caffeZoom2d(conv4, zoom_factor=2), conv2.shape)
        conv6 = caffeCrop2d_as(caffeZoom2d(conv6, zoom_factor=4), conv2.shape)
        conv7 = self.conv7(torch.cat([x, conv2, conv4, conv6], dim=1))
        return torch.cat((x, conv7), dim=1)