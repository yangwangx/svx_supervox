import torch
import smear_3d_cuda

__all__ = ['smear']

class Smear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spFeat, spIndx):
        # check
        # forward
        img_spFeat = smear_3d_cuda.forward(spFeat, spIndx)
        # context
        variables = [spIndx,]
        ctx.save_for_backward(*variables)
        ctx.K = spFeat.shape[-1]
        return img_spFeat

    @staticmethod
    def backward(ctx, grad_img_spFeat):
        # context
        spIndx, = ctx.saved_variables
        K = ctx.K
        # backward
        grad_spFeat = smear_3d_cuda.backward(grad_img_spFeat.contiguous(), spIndx, K)
        return grad_spFeat, None

smear = Smear.apply