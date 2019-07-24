import numpy as np, random
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as FF

__all__ = ['AverageMeter', 'AverageMeters', 'accuracy',
           'pairwise_distances_row', 'pairwise_distances_col',
           'batch_pairwise_distances_row', 'batch_pairwise_distances_col']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def AverageMeters(keys):
    """Create a dictionary of AverageMeters"""
    AMs = edict()
    for key in keys:
        AMs[key] = AverageMeter()
    return AMs

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    Args: 
        output:  the predicted class-wise scores, torch tensor of shape (B, C)
        target:  the ground-truth class labels, torch tensor of shape (B,)
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def pairwise_distances_row(x, y=None):
    """ Computes the pairwise euclidean distances between rows of x and rows of y.
    Args:
        x: torch tensor of shape (m, d)
        y: torch tensor of shape (n, d), or None
    Returns:
        dist: torch tensor of shape (m, n), or (m, m)
    """
    x_norm = (x**2).sum(dim=1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def pairwise_distances_col(x, y=None):
    """ Computes the pairwise euclidean distances between cols of x and cols of y.
    Args:
        x: torch tensor of shape (d, m)
        y: torch tensor of shape (d, n), or None
    Returns:
        dist: torch tensor of shape (m, n), or (m, m)
    """
    x_norm = (x**2).sum(dim=0).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(dim=0).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(torch.transpose(x, 0, 1), y)
    return dist    

def batch_pairwise_distances_row(x, y=None):
    """ Computes the pairwise euclidean distances between rows of x and rows of y.
    Args:
        x: torch tensor of shape (b, m, d)
        y: torch tensor of shape (b, n, d), or None
    Returns:
        dist: torch tensor of shape (b, m, n), or (b, m, m)
    """
    B, M, _ = x.shape
    x_norm = (x**2).sum(dim=2).view(B, -1, 1)
    if y is not None:
        y_norm = (y**2).sum(dim=2).view(B, 1, -1)
    else:
        y = x
        y_norm = x_norm.view(B, 1, -1)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
    return dist

def batch_pairwise_distances_col(x, y=None):
    """ Computes the pairwise euclidean distances between cols of x and cols of y.
    Args:
        x: torch tensor of shape (b, d, m)
        y: torch tensor of shape (b, d, n), or None
    Returns:
        dist: torch tensor of shape (b, m, n), or (b, m, m)
    """
    B, _, M = x.shape
    x_norm = (x**2).sum(dim=1).view(B, -1, 1)
    if y is not None:
        y_norm = (y**2).sum(dim=1).view(B, 1, -1)
    else:
        y = x
        y_norm = x_norm.view(B, 1, -1)
    dist = x_norm + y_norm - 2.0 * torch.bmm(torch.transpose(x, 1, 2), y)
    return dist
