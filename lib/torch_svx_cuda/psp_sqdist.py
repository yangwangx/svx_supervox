import torch
import psp_sqdist_3d_cuda

__all__ = ['compute_sqdist']

class PSP_sqdist(torch.autograd.Function):
    """computes pairwise eucledian squared distance between pixels and surrounding superpixels.
    Args:
        pFeat is of size BxCxLxHxW  - pixel feature
        spFeat is of size BxCxK - spixel feature
        init_spIndx is of size Bx1xLxHxW - initial spixel index
    Returns:
        sqdist is of size Bx27xLxHxW - pairwise distance between pixels and surrounding superpixels
    """
    @staticmethod
    def forward(ctx, pFeat, spFeat, init_spIndx, Kl, Kh, Kw):
        # check
        B, C, L, H, W = pFeat.shape
        sB, sC, K = spFeat.shape
        iB, iC, _, _, _ = init_spIndx.shape
        assert sB == B and sC == C and iB == B and iC == 1
        assert K == Kl * Kh * Kw
        # forward
        sqdist = psp_sqdist_3d_cuda.forward(pFeat, spFeat, init_spIndx, Kl, Kh, Kw)
        # context
        variables = [pFeat, spFeat, init_spIndx]
        ctx.save_for_backward(*variables)
        ctx.Kl, ctx.Kh, ctx.Kw = Kl, Kh, Kw
        return sqdist

    @staticmethod
    def backward(ctx, grad_sqdist):
        # context
        pFeat, spFeat, init_spIndx = ctx.saved_variables
        Kl, Kh, Kw = ctx.Kl, ctx.Kh, ctx.Kw
        # backward
        grad_pFeat, grad_spFeat = psp_sqdist_3d_cuda.backward(grad_sqdist.contiguous(),
            pFeat, spFeat, init_spIndx, Kl, Kh, Kw)
        return grad_pFeat, grad_spFeat, None, None, None, None

compute_sqdist = PSP_sqdist.apply