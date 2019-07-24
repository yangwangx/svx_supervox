import torch
import hierFeat_cuda

__all__ = ['hierFeat_collect']

class HierFeat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spFeat, spSize, assign, K):
        # forward
        hierFeat, hierSize = hierFeat_cuda.forward(spFeat, spSize, assign, K)
        return hierFeat, hierSize

    @staticmethod
    def backward(ctx, grad_hierFeat, grad_hierSize):
        return None, None, None, None

hierFeat_collect = HierFeat.apply