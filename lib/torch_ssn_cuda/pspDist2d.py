import torch
import pspDist2d_cuda

__all__ = ['pspDist2d']

class PSPDist2d(torch.autograd.Function):
    """computes pairwise eucledian squared distance between pixels and surrounding superpixels.
    Args:
        pFeat is of size BxCxHxW  - pixel feature
        spFeat is of size BxCxK - spixel feature
        init_spIndx is of size Bx1xHxW - initial spixel index
    Returns:
        sqdist is of size Bx9xHxW - pairwise distance between pixels and surrounding superpixels
    """
    @staticmethod
    def forward(ctx, pFeat, spFeat, init_spIndx, Kh, Kw):
        # check
        B, C, H, W = pFeat.shape
        sB, sC, K = spFeat.shape
        iB, iC, _, _ = init_spIndx.shape
        assert sB == B and sC == C and iB == B and iC == 1
        assert K == Kh * Kw
        # forward
        sqdist = pspDist2d_cuda.forward(pFeat, spFeat, init_spIndx, Kh, Kw)
        # context
        variables = [pFeat, spFeat, init_spIndx]
        ctx.save_for_backward(*variables)
        ctx.Kh, ctx.Kw = Kh, Kw
        return sqdist

    @staticmethod
    def backward(ctx, grad_sqdist):
        # context
        pFeat, spFeat, init_spIndx = ctx.saved_variables
        Kh, Kw = ctx.Kh, ctx.Kw
        # backward
        grad_pFeat, grad_spFeat = pspDist2d_cuda.backward(grad_sqdist.contiguous(),
            pFeat, spFeat, init_spIndx, Kh, Kw)
        return grad_pFeat, grad_spFeat, None, None, None

pspDist2d = PSPDist2d.apply