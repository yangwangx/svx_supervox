import torch
import spFeatUpdate_3d_cuda

__all__ = ['SpixelFeature_update']

class SPFeatUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pFeat, assoc, init_spIndx, Kl, Kh, Kw):
        # check
        B, C, L, H, W = pFeat.shape
        B2, C2, _, _, _ = assoc.shape  # B 27 L H W
        B3, C3, _, _, _ = init_spIndx.shape  # B 1 L H W
        assert B2 == B and B3 == B and C2 == 27 and C3 == 1
        # forward
        spFeat, spWght, = spFeatUpdate_3d_cuda.forward(pFeat, assoc, init_spIndx, Kl, Kh, Kw)
        # context
        variables = [spFeat, spWght, pFeat, assoc, init_spIndx]
        ctx.save_for_backward(*variables)
        ctx.Kl, ctx.Kh, ctx.Kw = Kl, Kh, Kw
        return spFeat

    @staticmethod
    def backward(ctx, grad_spFeat):
        # context
        spFeat, spWght, pFeat, assoc, init_spIndx = ctx.saved_variables
        Kl, Kh, Kw = ctx.Kl, ctx.Kh, ctx.Kw
        # backward
        grad_pFeat, grad_assoc = spFeatUpdate_3d_cuda.backward(grad_spFeat.contiguous(), 
            spFeat, spWght, pFeat, assoc, init_spIndx, Kl, Kh, Kw)
        return grad_pFeat, grad_assoc, None, None, None, None

SpixelFeature_update = SPFeatUpdate.apply