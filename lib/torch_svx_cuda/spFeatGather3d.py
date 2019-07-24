import torch
import spFeatGather3d_cuda

__all__ = ['spFeatGather3d']

class SPFeatGather3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pFeat, init_spIndx, num_spixels, ignore_idx_value=-10, ignore_feature_value=255):
        # check
        B, C, L, H, W = pFeat.shape
        B2, C2, L2, H2, W2 = init_spIndx.shape
        assert B2 == B and C2 == 1
        # forward
        spFeat, spSize = spFeatGather3d_cuda.forward(pFeat, init_spIndx, num_spixels, ignore_idx_value, ignore_feature_value)
        return spFeat, spSize

    @staticmethod
    def backward(ctx, grad_spFeat, grad_spSize):
        return None, None, None, None, None

spFeatGather3d = SPFeatGather3d.apply
