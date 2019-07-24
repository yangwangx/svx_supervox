import torch
import torch.nn as nn
import torch.nn.functional as FF
from .torch_svx_cuda import *

__all__ = ['get_pFeat_yxlab', 'get_pFeat_tyxlab',  
           'SVX', 'SVX_hier', 'compute_loss']

def get_pFeat_yxlab(vid_lab, yx_scale, lab_scale):
    """combines position (YX), and color (LAB) channels
    Args:
        vid_lab: tensor of shape (B,3,L,H,W) with "LAB" color channels
    """
    B, C, L, H, W = vid_lab.shape
    Y = torch.arange(H, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, 1, H, 1).expand(B, 1, L, H, W)
    X = torch.arange(W, device=vid_lab.device, dtype=vid_lab.dtype).view(1, 1, 1, 1, W).expand(B, 1, L, H, W)
    pFeat_yxlab = torch.cat([yx_scale*Y, yx_scale*X, lab_scale*vid_lab], dim=1)
    return pFeat_yxlab

def get_pFeat_tyxlab(vid_lab, t_scale, yx_scale, lab_scale):
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

def compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=-1.0):
    return torch.softmax(scaling * compute_sqdist(pFeat, spFeat, init_spIndx, Kl, Kh, Kw), dim=1)  # Bx27xLxHxW

def compute_final_spixel_labels(psp_assoc, init_spIndx, Kl, Kh, Kw):
    relIndx = torch.argmax(psp_assoc.detach(), dim=1, keepdim=True)  # Bx1xLxHxW
    absIndx = relToAbsIndex(relIndx.float(), init_spIndx, Kl, Kh, Kw)  # Bx1xLxHxW
    return absIndx

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

def soft_smear(spFeat, psp_assoc, init_spIndx, Kl, Kh, Kw):
    # concatenate neighboring supervoxel features
    B, C, K = spFeat.shape  # B C K
    _, _, L, H, W = psp_assoc.shape  # B 27 L H W
    spFeat = spFeat.view(B, C, Kl, Kh, Kw)  # B C Kl Kh Kw
    W_concat = get_W_concat_3x3x3(C).to(spFeat.device)
    spFeat_concat = FF.conv3d(spFeat, W_concat, bias=None, stride=1, padding=1, groups=C)
    spFeat_concat = spFeat_concat.view(B, C*27, K)  # B C27 K
    # spread features to pixels
    img_spFeat_concat = smear(spFeat_concat, init_spIndx)  # B C27 L H W
    # weighted sum
    img_spFeat_concat = img_spFeat_concat.view(B, C, 27, L, H, W)
    psp_assoc = psp_assoc.view(B, 1, 27, L, H, W)
    recon_feat = (img_spFeat_concat * psp_assoc).sum(dim=2)  # B C L H W
    return recon_feat

def euclidean_loss(x, y):
    B, C, L, H, W = x.shape
    loss = ((x - y) ** 2).sum() / (2.0 * B * L * H * W)
    return loss

def position_color_loss(recon, pFeat):
    pos_recon, col_recon = recon[:, :3], recon[:, 3:]
    pos_pFeat, col_pFeat = pFeat[:, :3], pFeat[:, 3:]
    pos_loss = euclidean_loss(pos_recon, pos_pFeat)
    color_loss = euclidean_loss(col_recon, col_pFeat)
    return pos_loss, color_loss

# create_ssn_net
class SVX(nn.Module):
    def __init__(self, use_cnn=True, num_in=6, num_out=14, num_ch=32, softscale=-1.0):
        super(SVX, self).__init__()
        self.num_steps = None
        self.Kl = None
        self.Kh = None
        self.Kw = None
        self.t_scale = None
        self.yx_scale = None
        self.lab_scale = None
        self.softscale = softscale
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = SVX_CNN(num_in=num_in, num_out=num_out, num_ch=num_ch)
        else:
            self.cnn = None

    def configure(self, vid_shape, Kl, Khw, p_scale, lab_scale, num_steps):
        _, _, L, H, W = vid_shape
        Kh = int(np.floor(np.sqrt(float(Khw) * H / W)))
        Kw = int(np.floor(np.sqrt(float(Khw) * W / H)))
        Khw = int(Kl * Kw)
        Klhw = int(Kl * Khw)
        yx_scale_l = Kl / (p_scale * L)
        yx_scale_h = Kh / (p_scale * H)
        yx_scale_w = Kw / (p_scale * W)
        yx_scale = max(yx_scale_h, yx_scale_w)
        self.num_steps = int(num_steps)
        self.Kl = int(Kl)
        self.Kh = int(Kh)
        self.Kw = int(Kw)
        self.t_scale = float(yx_scale_l)
        self.yx_scale = float(yx_scale)
        self.lab_scale = float(lab_scale)
        return L, H, W, Kl, Kh, Kw, Khw, Klhw

    def forward(self, vid_lab, init_spIndx):
        Kl, Kh, Kw = self.Kl, self.Kh, self.Kw
        # compute pixel features (TYXLAB)
        pFeat_tyxlab = get_pFeat_tyxlab(vid_lab,
            t_scale=self.t_scale, yx_scale=self.yx_scale, lab_scale=self.lab_scale)
        if self.use_cnn:
            pFeat = torch.cat((pFeat_tyxlab, self.cnn(pFeat_tyxlab)), dim=1)
        else:
            pFeat = pFeat_tyxlab
        # initial superpixel features
        spFeat, _ = SpixelFeature(pFeat, init_spIndx, Kl*Kh*Kw)  # BxCxK
        # multiple iteration
        for i in range(1, self.num_steps):
            # compute pixel-superpixel assignments
            psp_assoc = compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)  # Bx27xLxHxW
            # compute superpixel features
            spFeat, _ = SpixelFeature_update(pFeat, psp_assoc, init_spIndx, Kl, Kh, Kw)  # BxCx1x1xK
        # compute final_psp_assoc
        psp_assoc = compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)  # Bx27xLxHxW
        final_spIndx = compute_final_spixel_labels(psp_assoc, init_spIndx, Kl, Kh, Kw)
        return pFeat, spFeat, psp_assoc, final_spIndx

def compute_loss(pFeat, final_psp_assoc, init_spIndx, final_spIndx, onehot, Kl, Kh, Kw):
    pFeat_tyxlab = pFeat[:, :6].contiguous()
    # cycle loss for position
    spFeat_tyxlab = SpixelFeature_update(pFeat_tyxlab, final_psp_assoc, init_spIndx, Kl, Kh, Kw)
    recon_tyxlab = smear(spFeat_tyxlab, final_spIndx.float())
    loss_pos, loss_col = position_color_loss(recon_tyxlab, pFeat_tyxlab)
    # cycle loss for label
    spLabel = SpixelFeature_update(onehot, final_psp_assoc, init_spIndx, Kl, Kh, Kw)    
    # Convert spixel labels back to pixel labels
    recon_label = soft_smear(spLabel, final_psp_assoc, init_spIndx, Kl, Kh, Kw)
    recon_label = relu_L1(recon_label)  # B 2 L H W
    B, C, L, H, W = recon_label.shape
    loss_label = FF.kl_div((recon_label+1e-8).log(), onehot, reduction='batchmean') / (L * H * W)
    return loss_pos, loss_col, loss_label

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

def hierSVX(spFeat, spSize, r=0.5, unfold=8):
    with torch.no_grad():
        B, C, K = spFeat.shape
        hier_K = int(r * K)
        hier_spFeat = init_kmeans_feature(spFeat, hier_K)
        # hard kmeans
        for i in range(unfold):
            hier_assign = torch.argmin(batch_pairwise_distances_col(spFeat, hier_spFeat), dim=2).float()
            hier_spFeat, hier_spSize = hierFeat_collect(spFeat, spSize, hier_assign, hier_K)
    return hier_assign, hier_spFeat, hier_spSize

# create_ssn_net
class SVX_hier(SVX):
    def __init__(self, use_cnn=True, num_in=6, num_out=14, num_ch=32, softscale=-1.0, ratio=0.5):
        super(SVX_hier, self).__init__(use_cnn=True, num_in=6, num_out=14, num_ch=32, softscale=-1.0)
        self.ratio = ratio

    def forward(self, vid_lab, init_spIndx):
        Kl, Kh, Kw = self.Kl, self.Kh, self.Kw
        # compute pixel features (TYXLAB)
        pFeat_tyxlab = get_pFeat_tyxlab(vid_lab,
            t_scale=self.t_scale, yx_scale=self.yx_scale, lab_scale=self.lab_scale)
        if self.use_cnn:
            pFeat = torch.cat((pFeat_tyxlab, self.cnn(pFeat_tyxlab)), dim=1)
        else:
            pFeat = pFeat_tyxlab
        B, C, L, H, W = pFeat.shape
        # initial superpixel features
        spFeat, _ = SpixelFeature(pFeat, init_spIndx, Kl*Kh*Kw)  # BxCxK
        # multiple iteration
        for i in range(1, self.num_steps):
            # compute pixel-superpixel assignments
            psp_assoc = compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)  # Bx27xLxHxW
            # compute superpixel features
            spFeat = SpixelFeature_update(pFeat, psp_assoc, init_spIndx, Kl, Kh, Kw)  # BxCx1x1xK
        # compute final_psp_assoc
        psp_assoc = compute_psp_assoc(pFeat, spFeat, init_spIndx, Kl, Kh, Kw, scaling=self.softscale)  # Bx27xLxHxW
        final_spIndx = compute_final_spixel_labels(psp_assoc, init_spIndx, Kl, Kh, Kw)   
        with torch.no_grad():
            # hier kmeans
            K = Kl*Kh*Kw
            spFeat, spSize = SpixelFeature(pFeat, final_spIndx.float(), K)
            spFeat = spFeat.view(B, C, K)
            spSize = spSize.view(B, K)
            hier_assign, _, _ = hierSVX(spFeat, spSize, r=self.ratio, unfold=8)
            # final_spIndx: B 1 L H W
            # hier_assign:  B K
            hier_assign = hier_assign.long().view(B, K, 1, 1, 1).expand(B, K, L, H, W)
            hier_spIndx = torch.gather(input=hier_assign, dim=1, index=final_spIndx.long())
        return None, None, None, hier_spIndx.long()