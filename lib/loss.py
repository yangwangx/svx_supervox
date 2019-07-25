import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF
from .torch_ssn_cuda import *
from .torch_svx_cuda import *

__all__ = ['relu_L1', 'get_W_concat_3x3', 'get_W_concat_3x3x3',
           'soft_smear_2d', 'soft_smear_3d', 'compute_svx_loss']

def relu_L1(data, dim=1):
    """ applies relu and l1_norm."""
    data_relu = FF.relu(data)
    l1norm = data_relu.sum(dim=dim, keepdim=True)
    return data_relu / (l1norm + 1e-12)

def get_W_concat_3x3(C):
    W_concat = torch.zeros(9*C, 1, 3, 3, requires_grad=False)
    for j in range(C):
        for i in range(9):
            W_concat.data[9 * j + i, 0, i // 3, i % 3] = 1.0
    return W_concat

def get_W_concat_3x3x3(C):
    W_concat = torch.zeros(27*C, 1, 3, 3, 3, requires_grad=False)
    for j in range(C):
        for i in range(27):
            W_concat.data[27 * j + i, 0, i//9, (i%9)//3, (i%9)%3] = 1.0
    return W_concat

def soft_smear_2d(spFeat, psp_assoc, init_spIndx, Kh, Kw):
    # concatenate neighboring superpixel features
    B, C, K = spFeat.shape  # B C K
    _, _, H, W = psp_assoc.shape  # B 9 H W
    spFeat = spFeat.view(B, C, Kh, Kw)  # B C Kh Kw
    W_concat = get_W_concat_3x3(C).to(spFeat.device)
    spFeat_concat = FF.conv2d(spFeat, W_concat, bias=None, stride=1, padding=1, groups=C)
    spFeat_concat = spFeat_concat.view(B, C*9, 1, K)  # B C9 1 K
    # spread features to pixels
    pFeat_concat = spFeatSmear2d(spFeat_concat, init_spIndx).view(B, C, 9, H, W)
    # weighted sum
    pFeat_recon = (pFeat_concat * psp_assoc.view(B, 1, 9, H, W)).sum(dim=2)  # B C H W
    return pFeat_recon

def soft_smear_3d(spFeat, psp_assoc, init_spIndx, Kl, Kh, Kw):
    # concatenate neighboring supervoxel features
    B, C, K = spFeat.shape  # B C K
    _, _, L, H, W = psp_assoc.shape  # B 27 L H W
    spFeat = spFeat.view(B, C, Kl, Kh, Kw)  # B C Kl Kh Kw
    W_concat = get_W_concat_3x3x3(C).to(spFeat.device)
    spFeat_concat = FF.conv3d(spFeat, W_concat, bias=None, stride=1, padding=1, groups=C)
    spFeat_concat = spFeat_concat.view(B, C*27, K)  # B C27 K
    # spread features to pixels
    pFeat_concat = spFeatSmear3d(spFeat_concat, init_spIndx)  # B C27 L H W
    # weighted sum
    pFeat_concat = pFeat_concat.view(B, C, 27, L, H, W)
    psp_assoc = psp_assoc.view(B, 1, 27, L, H, W)
    pFeat_recon = (pFeat_concat * psp_assoc).sum(dim=2)  # B C L H W
    return pFeat_recon

def compute_ssn_loss(pFeat_yxlab, final_psp_assoc, init_spIndx, final_spIndx, onehot, Kh, Kw):
    B, C, H, W = pFeat_yxlab.shape
    assert C == 5 and pFeat_yxlab.is_contiguous()
    # cycle loss for position and color
    spFeat_yxlab, _ = spFeatUpdate2d(pFeat_yxlab, final_psp_assoc, init_spIndx, Kh, Kw)
    recon_yxlab = spFeatSmear2d(spFeat_yxlab, final_spIndx.float())
    pFeat_yx, pFeat_lab = pFeat_yxlab[:, :2], pFeat_yxlab[:, 2:]
    recon_yx, recon_lab = recon_yxlab[:, :2], recon_yxlab[:, 2:]
    _euclidean = lambda x, y: ((x - y) ** 2).mean() * (x.shape[1] / 2.0)
    loss_pos = _euclidean(recon_yx, pFeat_yx)
    loss_col = _euclidean(recon_lab, pFeat_lab)
    # cycle loss for label
    spLabel, _ = spFeatUpdate2d(onehot, final_psp_assoc, init_spIndx, Kh, Kw)
    recon_label = soft_smear_2d(spLabel, final_psp_assoc, init_spIndx, Kh, Kw)
    recon_label = relu_L1(recon_label)
    loss_label = FF.kl_div((recon_label+1e-8).log(), onehot, reduction='batchmean') / (H * W)
    return loss_pos, loss_col, loss_label

def compute_svx_loss(pFeat_tyxlab, final_psp_assoc, init_spIndx, final_spIndx, onehot, Kl, Kh, Kw):
    B, C, L, H, W = pFeat_tyxlab.shape
    assert C == 6 and pFeat_tyxlab.is_contiguous()
    # cycle loss for position and color
    spFeat_tyxlab, _ = spFeatUpdate3d(pFeat_tyxlab, final_psp_assoc, init_spIndx, Kl, Kh, Kw)
    recon_tyxlab = spFeatSmear3d(spFeat_tyxlab, final_spIndx.float())
    pFeat_tyx, pFeat_lab = pFeat_tyxlab[:, :3], pFeat_tyxlab[:, 3:]
    recon_tyx, recon_lab = recon_tyxlab[:, :3], recon_tyxlab[:, 3:]
    _euclidean = lambda x, y: ((x - y) ** 2).mean() * (x.shape[1] / 2.0)
    loss_pos = _euclidean(recon_tyx, pFeat_tyx)
    loss_col = _euclidean(recon_lab, pFeat_lab)
    # cycle loss for label
    spLabel, _ = spFeatUpdate3d(onehot, final_psp_assoc, init_spIndx, Kl, Kh, Kw)    
    recon_label = soft_smear_3d(spLabel, final_psp_assoc, init_spIndx, Kl, Kh, Kw)
    recon_label = relu_L1(recon_label)
    loss_label = FF.kl_div((recon_label+1e-8).log(), onehot, reduction='batchmean') / (L * H * W)
    return loss_pos, loss_col, loss_label
