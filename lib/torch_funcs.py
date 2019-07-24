import torch
import torch.nn as nn
import torch.nn.functional as FF

__all__ = ['pairwise_distances_row', 'pairwise_distances_col',
           'batch_pairwise_distances_row', 'batch_pairwise_distances_col']

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