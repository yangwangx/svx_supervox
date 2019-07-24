import torch
import spFeat_3d_cuda

__all__ = ['SpixelFeature']

class SPFeat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pFeat, init_spIndx, num_spixels, ignore_idx_value=-10, ignore_feature_value=255.0):
        # check
        B, C, L, H, W = pFeat.shape
        B2, C2, _, _, _ = init_spIndx.shape
        assert B2 == B and C2 == 1
        # forward
        spFeat, spSize = spFeat_3d_cuda.forward(pFeat, init_spIndx, num_spixels, ignore_idx_value, ignore_feature_value)
        return spFeat, spSize  # B C K(=num_spixels)

    @staticmethod
    def backward(ctx, grad_spFeat, grad_spSize):
        return None, None, None, None, None

SpixelFeature = SPFeat.apply