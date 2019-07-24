import torch
import hierFeatGather_cuda

__all__ = ['hierFeatGather']

class HierFeatGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spFeat, spSize, assign, hierK):
        # forward
        hierFeat, hierSize = hierFeatGather_cuda.forward(spFeat, spSize, assign, hierK)
        return hierFeat, hierSize

    @staticmethod
    def backward(ctx, grad_hierFeat, grad_hierSize):
        return None, None, None, None

hierFeatGather = HierFeatGather.apply