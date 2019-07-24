import torch
import spFeatSmear2d_cuda

__all__ = ['spFeatSmear2d']

class SPFeatSmear2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spFeat, spIndx):
        # check
        # forward
        pFeat = spFeatSmear2d_cuda.forward(spFeat, spIndx)
        # context
        variables = [spIndx,]
        ctx.save_for_backward(*variables)
        ctx.K = spFeat.shape[-1]
        return pFeat

    @staticmethod
    def backward(ctx, grad_pFeat):
        # context
        spIndx, = ctx.saved_variables
        K = ctx.K
        # backward
        grad_spFeat = spFeatSmear2d_cuda.backward(grad_pFeat.contiguous(), spIndx, K)
        return grad_spFeat, None

spFeatSmear2d = SPFeatSmear2d.apply