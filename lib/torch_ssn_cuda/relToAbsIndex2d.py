import torch
import relToAbsIndex2d_cuda

__all__ = ['relToAbsIndex2d']

class RelToAbsIndex2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, relIndx, init_spIndx, Kh, Kw):
        absIndx = relToAbsIndex2d_cuda.forward(relIndx, init_spIndx, Kh, Kw)
        return absIndx

    @staticmethod
    def backward(ctx, grad_absIndx):
        return None, None, None, None

relToAbsIndex2d = RelToAbsIndex2d.apply