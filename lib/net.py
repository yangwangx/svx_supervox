import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF
from .torch_ssn_cuda import *
from .torch_svx_cuda import *
from .cnn_ssn import SSN_CNN
from .cnn_svx import SVX_CNN

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

class SSN(nn.Module):
    def __init__(self, use_cnn, num_in=5, num_out=15, num_ch=64, pretrained_npz=''):
        super(SSN, self).__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = None
        else:
            self.cnn = SSN_CNN(num_in=num_in, num_out=num_out, num_ch=num_ch, pretrained_npz=pretrained_npz)

    def configure(self, img_shape, n_sv, p_scale, lab_scale, softscale, num_steps):
        _, _, H, W = img_shape
        Kh = int(np.floor(np.sqrt(float(n_sv) * H / W)))
        Kw = int(np.floor(np.sqrt(float(n_sv) * W / H)))
        K = Kh * Kw
        y_scale = Kh / (p_scale * H)
        x_scale = Kw / (p_scale * W)
        yx_scale = max(y_scale, x_scale)
        self.Kh = int(Kh)
        self.Kw = int(Kw)
        self.yx_scale = float(yx_scale)
        self.lab_scale = float(lab_scale)
        self.softscale = float(softscale)
        self.num_steps = int(num_steps)
        return H, W, Kh, Kw, K

    def forward(self, img_lab, init_spIndx):
        Kh, Kw = self.Kh, self.Kw
        pFeat_yxlab = get_pFeat_yxlab_2d(img_lab, yx_scale=self.yx_scale, lab_scale=self.lab_scale)
        if self.use_cnn:
            pFeat = torch.cat((pFeat_yxlab, self.cnn(pFeat_yxlab)), dim=1)
        else:
            pFeat = pFeat_yxlab
        spFeat, _ = spFeatGather2d(pFeat, init_spIndx, Kh*Kw)
        for _ in range(1, self.num_steps):
            psp_assoc = compute_psp_assoc_2d(pFeat, spFeat, init_spIndx, Kh, Kw, scaling=self.softscale)
            spFeat, _ = spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
        # compute final psp_assoc
        psp_assoc = compute_psp_assoc_2d(pFeat, spFeat, init_spIndx, Kh, Kw, scaling=self.softscale)
        final_spIndx = compute_final_spixel_index_2d(psp_assoc, init_spIndx, Kh, Kw)
        return pFeat, spFeat, psp_assoc, final_spIndx

class SVX(nn.Module):
    def __init__(self, use_cnn=True, num_in=6, num_out=14, num_ch=32):
        super(SVX, self).__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = SVX_CNN(num_in=num_in, num_out=num_out, num_ch=num_ch)
        else:
            self.cnn = None

    def configure(self, vid_shape, t_sv, n_sv, p_scale, lab_scale, softscale, num_steps):
        _, _, L, H, W = vid_shape
        Kl = int(t_sv)
        Kh = int(np.floor(np.sqrt(float(n_sv) * H / W)))
        Kw = int(np.floor(np.sqrt(float(n_sv) * W / H)))
        Khw = Kl * Kw
        Klhw = Kl * Khw
        t_scale = Kl / (p_scale * L)
        y_scale = Kh / (p_scale * H)
        x_scale = Kw / (p_scale * W)
        yx_scale = max(y_scale, x_scale)        
        self.Kl = int(Kl)
        self.Kh = int(Kh)
        self.Kw = int(Kw)
        self.t_scale = float(t_scale)
        self.yx_scale = float(yx_scale)
        self.lab_scale = float(lab_scale)
        self.softscale = float(softscale)
        self.num_steps = int(num_steps)
        return L, H, W, Kl, Kh, Kw, Khw, Klhw

    def forward(self, vid_lab, init_spIndx):
        Kl, Kh, Kw = self.Kl, self.Kh, self.Kw
        pFeat_tyxlab = get_pFeat_tyxlab_3d(vid_lab, t_scale=self.t_scale, 
                                                    yx_scale=self.yx_scale,
                                                    lab_scale=self.lab_scale)
        if self.use_cnn:
            pFeat = torch.cat((pFeat_tyxlab, self.cnn(pFeat_tyxlab)), dim=1)
        else:
            pFeat = pFeat_tyxlab
        spFeat, _ = spFeatGather3d(pFeat, init_spIndx, Kl*Kh*Kw)
        for _ in range(1, self.num_steps):
            psp_assoc = compute_psp_assoc_3d(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)
            spFeat, _ = spFeatUpdate3d(pFeat, psp_assoc, init_spIndx, Kl, Kh, Kw)
        # compute final psp_assoc
        psp_assoc = compute_psp_assoc_3d(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)
        final_spIndx = compute_final_spixel_index_3d(psp_assoc, init_spIndx, Kl, Kh, Kw)
        return pFeat, spFeat, psp_assoc, final_spIndx

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