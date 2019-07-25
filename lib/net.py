import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF
from .torch_ssn_cuda import *
from .torch_svx_cuda import *

from skimage.segmentation import mark_boundaries
def get_spixel_image(img, spix_index):
    """marks superpixel boundaries on the image"""
    spixel_image = mark_boundaries(img / 255.0, spix_index.astype(int), color=(1, 0, 0))
    spixel_image = (spixel_image * 255.0).astype(np.uint8)
    return spixel_image

from scipy import interpolate
def get_spixel_init(num_spixels, img_height, img_width):
    """computes initial superpixel index map"""
    k_w = int(np.floor(np.sqrt(num_spixels * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(num_spixels * img_height / img_width)))

    spixel_height = img_height / k_h
    spixel_width = img_width / k_w

    h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1, spixel_height)
    w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1, spixel_width)
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height, img_width))  # H W
    return spixel_initmap, k_h, k_w

def get_pFeat_yxlab_2d(img_lab, yx_scale, lab_scale):
    """combines position (YX), and color (LAB) channels
    Args:
        img_lab: tensor of shape (B,3,H,W) with "LAB" color channels
    """
    B, C, H, W = img_lab.shape
    Y = torch.arange(H, device=img_lab.device, dtype=img_lab.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    X = torch.arange(W, device=img_lab.device, dtype=img_lab.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    pFeat_yxlab = torch.cat([yx_scale*Y, yx_scale*X, lab_scale*img_lab], dim=1)
    return pFeat_yxlab

def get_pFeat_yxlab_3d(vid_lab, yx_scale, lab_scale):
    """combines position (YX), and color (LAB) channels
    Args:
        vid_lab: tensor of shape (B,3,L,H,W) with "LAB" color channels
    """
    B, C, L, H, W = vid_lab.shape
    Y = torch.arange(H, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, 1, H, 1).expand(B, 1, L, H, W)
    X = torch.arange(W, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, 1, 1, W).expand(B, 1, L, H, W)
    pFeat_yxlab = torch.cat([yx_scale*Y, yx_scale*X, lab_scale*vid_lab], dim=1)
    return pFeat_yxlab

def get_pFeat_tyxlab_3d(vid_lab, t_scale, yx_scale, lab_scale):
    """combines time (T), position (YX), and color (LAB) channels
    Args:
        vid_lab: tensor of shape (B,3,L,H,W) with "LAB" color channels
    """
    B, C, L, H, W = vid_lab.shape
    T = torch.arange(L, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, L, 1, 1).expand(B, 1, L, H, W)
    Y = torch.arange(H, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, 1, H, 1).expand(B, 1, L, H, W)
    X = torch.arange(W, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, 1, 1, W).expand(B, 1, L, H, W)
    pFeat_tyxlab = torch.cat([t_scale*T, yx_scale*Y, yx_scale*X, lab_scale*vid_lab], dim=1)
    return pFeat_tyxlab

def compute_psp_assoc_2d(pFeat, spFeat, init_spIndx, Kh, Kw, scaling=-1.0):
    return torch.softmax(scaling * pspDist2d(pFeat, spFeat, init_spIndx, Kh, Kw), dim=1)  # Bx9xHxW

def compute_psp_assoc_3d(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=-1.0):
    return torch.softmax(scaling * pspDist3d(pFeat, spFeat, init_spIndx, Kl, Kh, Kw), dim=1)  # Bx27xLxHxW

def compute_final_spixel_index_2d(psp_assoc, init_spIndx, Kh, Kw):
    relIndx = torch.argmax(psp_assoc, dim=1, keepdim=True)  # Bx1xHxW
    absIndx = relToAbsIndex2d(relIndx.float(), init_spIndx, Kh, Kw)  # Bx1xHxW
    return absIndx

def compute_final_spixel_index_3d(psp_assoc, init_spIndx, Kl, Kh, Kw):
    relIndx = torch.argmax(psp_assoc, dim=1, keepdim=True)  # Bx1xLxHxW
    absIndx = relToAbsIndex3d(relIndx.float(), init_spIndx, Kl, Kh, Kw)  # Bx1xLxHxW
    return absIndx

def init_kmeans_feature(spFeat, hier_K):
    with torch.no_grad():
        B, C, K = spFeat.shape
        spDist = batch_pairwise_distances_col(spFeat)  # B K K
        hier_spFeat = torch.zeros(B, C, hier_K, dtype=spFeat.dtype, device=spFeat.device)  # B C hier_K
        for b in range(B):
            idxs = [random.randrange(K), ]
            min_dist = spDist[b, idxs[-1]]
            for k in range(1, hier_K):
                idxs.append(torch.argmax(min_dist).item())
                min_dist = torch.min(min_dist, spDist[b, idxs[-1]])
            idxs = torch.LongTensor(sorted(idxs)).to(spFeat.device)
            hier_spFeat[b] = spFeat[b, :, idxs]
    return hier_spFeat  # B C hier_K

def hierSSN(spFeat, spSize, r=0.5, unfold=8):
    with torch.no_grad():
        B, C, K = spFeat.shape
        hier_K = int(r * K)
        hier_spFeat = init_kmeans_feature(spFeat, hier_K)
        # hard kmeans
        for i in range(unfold):
            hier_assign = torch.argmin(batch_pairwise_distances_col(spFeat, hier_spFeat), dim=2).float()
            hier_spFeat, hier_spSize = hierFeat_collect(spFeat, spSize, hier_assign, hier_K)
    return hier_assign, hier_spFeat, hier_spSize

def batch_pairwise_distances_col(x, y=None):
    """ Computes the pairwise euclidean distances between cols of x and cols of y.
    Args:
        x: torch tensor of shape (b, d, m)
        y: torch tensor of shape (b, d, n), or None
    Returns:
        dist: torch tensor of shape (b, m, n), or (b, m, m)
    """
    B, _, M = x.shape
    x_norm = (x**2).sum(dim=1).view(B, -1, 1)
    if y is not None:
        y_norm = (y**2).sum(dim=1).view(B, 1, -1)
    else:
        y = x
        y_norm = x_norm.view(B, 1, -1)
    dist = x_norm + y_norm - 2.0 * torch.bmm(torch.transpose(x, 1, 2), y)
    return dist

def init_kmeans_feature(spFeat, hier_K):
    with torch.no_grad():
        B, C, K = spFeat.shape
        spDist = batch_pairwise_distances_col(spFeat)  # B K K
        hier_spFeat = torch.zeros(B, C, hier_K, dtype=spFeat.dtype, device=spFeat.device)  # B C hier_K
        for b in range(B):
            idxs = [random.randrange(K), ]
            min_dist = spDist[b, idxs[-1]]
            for k in range(1, hier_K):
                idxs.append(torch.argmax(min_dist).item())
                min_dist = torch.min(min_dist, spDist[b, idxs[-1]])
            idxs = torch.LongTensor(sorted(idxs)).to(spFeat.device)
            hier_spFeat[b] = spFeat[b, :, idxs]
    return hier_spFeat  # B C hier_K

def hard_kmeans(spFeat, spSize, spIndx, ratio=0.5, step=10):
    B, C, K = spFeat.shape
    hier_K = int(ratio * K)
    if spIndx.dim() == 4:
        _, _, H, W = spIndx.shape
        mode = '2d'
    elif spIndx.dim() == 5:
        _, _, L, H, W = spIndx.shape
        mode = '3d'
    else:
        raise NotImplementedError
    with torch.no_grad():
        hier_spFeat = init_kmeans_feature(spFeat, hier_K)
        for _ in range(step):
            hier_assign = torch.argmin(batch_pairwise_distances_col(spFeat, hier_spFeat), dim=2)
            hier_spFeat, hier_spSize = hierFeatGather(spFeat, spSize, hier_assign.float(), hier_K)
        if mode == '2d':
            hier_assign = hier_assign.view(B, K, 1, 1).expand(B, K, H, W)
        elif mode == '3d':
            hier_assign = hier_assign.view(B, K, 1, 1, 1).expand(B, K, L, H, W)
        hier_spIndx = torch.gather(input=hier_assign, dim=1, index=spIndx.long())
    return hier_spFeat, hier_spSize, hier_spIndx


# create_ssn_net
# class SVX(nn.Module):
#     def __init__(self, use_cnn=True, num_in=6, num_out=14, num_ch=32, softscale=-1.0):
#         super(SVX, self).__init__()
#         self.num_steps = None
#         self.Kl = None
#         self.Kh = None
#         self.Kw = None
#         self.t_scale = None
#         self.yx_scale = None
#         self.lab_scale = None
#         self.softscale = softscale
#         self.use_cnn = use_cnn
#         if use_cnn:
#             self.cnn = SVX_CNN(num_in=num_in, num_out=num_out, num_ch=num_ch)
#         else:
#             self.cnn = None

#     def configure(self, vid_shape, Kl, Khw, p_scale, lab_scale, num_steps):
#         _, _, L, H, W = vid_shape
#         Kh = int(np.floor(np.sqrt(float(Khw) * H / W)))
#         Kw = int(np.floor(np.sqrt(float(Khw) * W / H)))
#         Khw = int(Kl * Kw)
#         Klhw = int(Kl * Khw)
#         yx_scale_l = Kl / (p_scale * L)
#         yx_scale_h = Kh / (p_scale * H)
#         yx_scale_w = Kw / (p_scale * W)
#         yx_scale = max(yx_scale_h, yx_scale_w)
#         self.num_steps = int(num_steps)
#         self.Kl = int(Kl)
#         self.Kh = int(Kh)
#         self.Kw = int(Kw)
#         self.t_scale = float(yx_scale_l)
#         self.yx_scale = float(yx_scale)
#         self.lab_scale = float(lab_scale)
#         return L, H, W, Kl, Kh, Kw, Khw, Klhw

#     def forward(self, vid_lab, init_spIndx):
#         Kl, Kh, Kw = self.Kl, self.Kh, self.Kw
#         # compute pixel features (TYXLAB)
#         pFeat_tyxlab = get_pFeat_tyxlab(vid_lab,
#             t_scale=self.t_scale, yx_scale=self.yx_scale, lab_scale=self.lab_scale)
#         if self.use_cnn:
#             pFeat = torch.cat((pFeat_tyxlab, self.cnn(pFeat_tyxlab)), dim=1)
#         else:
#             pFeat = pFeat_tyxlab
#         # initial superpixel features
#         spFeat, _ = spFeatGather3d(pFeat, init_spIndx, Kl*Kh*Kw)  # BxCxK
#         # multiple iteration
#         for i in range(1, self.num_steps):
#             # compute pixel-superpixel assignments
#             psp_assoc = compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)  # Bx27xLxHxW
#             # compute superpixel features
#             spFeat, _ = spFeatUpdate3d(pFeat, psp_assoc, init_spIndx, Kl, Kh, Kw)
#         # compute final_psp_assoc
#         psp_assoc = compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)  # Bx27xLxHxW
#         final_spIndx = compute_final_spixel_labels(psp_assoc, init_spIndx, Kl, Kh, Kw)
#         return pFeat, spFeat, psp_assoc, final_spIndx