import torch
import relToAbsIndex_3d_cuda

__all__ = ['relToAbsIndex']

class RelToAbsIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, relIndx, init_spIndx, Kl, Kh, Kw):
        absIndx = relToAbsIndex_3d_cuda.forward(relIndx, init_spIndx, Kl, Kh, Kw)
        return absIndx

    @staticmethod
    def backward(ctx, grad_absIndx):
        return None, None, None, None, None

relToAbsIndex = RelToAbsIndex.apply