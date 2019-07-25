import torch
import relToAbsIndex3d_cuda

__all__ = ['relToAbsIndex3d']

class RelToAbsIndex3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, relIndx, init_spIndx, Kl, Kh, Kw):
        absIndx = relToAbsIndex3d_cuda.forward(relIndx, init_spIndx, Kl, Kh, Kw)
        return absIndx

    @staticmethod
    def backward(ctx, grad_absIndx):
        return None, None, None, None, None

relToAbsIndex3d = RelToAbsIndex3d.apply