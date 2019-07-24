import torch
import spFeatUpdate2d_cuda

__all__ = ['spFeatUpdate2d']

class SPFeatUpdate2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pFeat, assoc, init_spIndx, Kh, Kw):
        # check
        B, C, H, W = pFeat.shape
        B2, C2, _, _ = assoc.shape  # B 9 H W
        B3, C3, _, _ = init_spIndx.shape  # B 1 H W
        assert B2 == B and B3 == B and C2 == 9 and C3 == 1
        # forward
        spFeat, spWght = spFeatUpdate2d_cuda.forward(pFeat, assoc, init_spIndx, Kh, Kw)
        # context
        variables = [spFeat, spWght, pFeat, assoc, init_spIndx]
        ctx.save_for_backward(*variables)
        ctx.Kh, ctx.Kw = Kh, Kw
        return spFeat, spWght

    @staticmethod
    def backward(ctx, grad_spFeat, grad_spWght):
        # context
        spFeat, spWght, pFeat, assoc, init_spIndx = ctx.saved_variables
        Kh, Kw = ctx.Kh, ctx.Kw
        # backward
        grad_pFeat, grad_assoc = spFeatUpdate2d_cuda.backward(grad_spFeat.contiguous(),
            spFeat, spWght, pFeat, assoc, init_spIndx, Kh, Kw)
        return grad_pFeat, grad_assoc, None, None, None

spFeatUpdate2d = SPFeatUpdate2d.apply