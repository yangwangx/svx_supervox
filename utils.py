import os, sys, glob, time, progressbar, argparse
from itertools import islice
from functools import partial
from easydict import EasyDict as edict
import random, math, numpy as np
import cv2, imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as FF
import torch.utils.data as DD
import torchvision
from tensorboardX import SummaryWriter
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    CWD = os.path.dirname(os.path.abspath(__file__)) + '/'
except NameError:
    CWD = ''
sys.path.append(CWD)
from lib import *

def get_empty_parser():
    return argparse.ArgumentParser()

def get_base_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--workers', default=4, type=int, help='number of workers for dataloader')
    parser.add_argument('--BtSz', default=8, type=int, help='batch size')
    parser.add_argument('--BtMerge', default=1, type=int, help='batch step for merged gradient update')
    parser.add_argument('--trRatio', default=1, type=float, help='ratio of training data used per epoch')
    parser.add_argument('--OneBatch', default=False, action='store_true', dest='OneBatch', help='debug with one batch')
    # model
    # loss
    # optimizer
    parser.add_argument('--noOptimizer', default=[], type=str, nargs='+', dest='noOptimizer')
    parser.add_argument('--solver', default='adam', choices=['adam','sgd'], help='which solver')
    parser.add_argument('--MM', default=0.9, type=float, help='momentum')
    parser.add_argument('--Beta', default=0.999, type=float, help='beta for adam')
    parser.add_argument('--WD', default=0, type=float, help='weight decay')
    # learning rate
    parser.add_argument('--LRPolicy', default='constant', type=str, choices=['constant', 'step', 'steps', 'exponential',], help='learning rate policy')
    parser.add_argument('--gamma', default=1.0, type=float, help='decay rate for learning rate')
    parser.add_argument('--LRStart', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--LRStep', default=10, type=int, help='steps to change learning rate')
    parser.add_argument('--LRSteps', default=[], type=int, nargs='+', dest='LRSteps', help='epochs before cutting learning rate')
    parser.add_argument('--nEpoch', default=3000, type=int, help='total epochs to run')
    # init & checkpoint
    parser.add_argument('--initModel', default='', help='init model in absence of checkpoint')
    parser.add_argument('--checkpoint', default=0, type=int, help='resume from checkpoint')
    # save & display
    parser.add_argument('--saveDir', default='results/default/', help='directory to save/log experiments')
    parser.add_argument('--saveStep', default=100, type=int, help='epoch step for snapshot')
    parser.add_argument('--evalStep', default=100, type=int, help='epoch step for evaluation')
    parser.add_argument('--dispIter', default=50, type=int, help='batch step for tensorboard')
    # other mode
    parser.add_argument('--evalMode', default=False, action='store_true', dest='evalMode', help='evaluation mode')
    parser.add_argument('--visMode', default=False, action='store_true', dest='visMode', help='visualization mode')
    parser.add_argument('--visDir', default='visual', type=str, help="dir to store visualization")
    parser.add_argument('--visNum', default=1000, type=int, help="number of videos to visualize")
    parser.add_argument('--applyMode', default=False, action='store_true', dest='applyMode', help='apply model to one gif')
    parser.add_argument('--applyFile', default='', type=str, help='path to gif')
    # misc
    parser.add_argument('--seed', default=1, type=int, help='random seed for torch/numpy')
    return parser

def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mkdir(dirname):
    os.makedirs(dirname, exist_ok=True)

def mkdir_save(state, state_file):
    mkdir(os.path.dirname(state_file))
    torch.save(state, state_file)

def print_options(opts, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opts).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    mkdir(opts.saveDir)
    file_name = os.path.join(opts.saveDir, 'opts_{}.txt'.format(time.asctime()))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
    return message

def create_optimizer(model, opts):
    optimizer = edict()
    if opts.solver == 'sgd':
        solver = partial(torch.optim.SGD, lr=opts.LRStart, momentum=opts.MM, weight_decay=opts.WD)
    elif opts.solver == 'adam':
        solver = partial(torch.optim.Adam, lr=opts.LRStart, betas=(opts.MM, opts.Beta), weight_decay=opts.WD)
    else:
        raise ValueError('optim solver "{}" not defined'.format(opts.solver))
    for key in model.keys():
        params = list(model[key].parameters())
        if params and key not in opts.noOptimizer: optimizer[key] = solver(params)
    return optimizer

def create_scheduler(optimizer, opts):
    scheduler = edict()
    for key in optimizer.keys():
        op = optimizer[key]
        if opts.LRPolicy == 'constant':
            scheduler[key] = torch.optim.lr_scheduler.ExponentialLR(op, gamma=1.0)
        elif opts.LRPolicy == 'step':
            scheduler[key] = torch.optim.lr_scheduler.StepLR(op, opts.LRStep, gamma=opts.gamma)
        elif opts.LRPolicy == 'steps':
            scheduler[key] = torch.optim.lr_scheduler.MultiStepLR(op, opts.LRSteps, gamma=opts.gamma)
        elif opts.LRPolicy == 'exponential':
            scheduler[key] = torch.optim.lr_scheduler.ExponentialLR(op, gamma=opts.gamma)
        else:
            raise ValueError('learning rate policy "{}" not defined'.format(opts.LRPolicy))
    return scheduler

def save_checkpoint(epoch, model, optimizer, opts):
    print('save model & optimizer @ epoch %d'%(epoch))
    ckpt_file = '%s/ckpt/ep-%04d.pt'%(opts.saveDir, epoch)
    state = {}
    # an error occurs when I use edict
    # possible reason: torch.optim.state_dict() has an 'state' attribute
    state['epoch'] = epoch
    for key in model.keys():
        _m = model[key]
        if hasattr(_m, 'module'): _m = _m.module
        state['model_'+key] = _m.state_dict()
    for key in optimizer.keys():
        state['optimizer_'+key] = optimizer[key].state_dict()
    mkdir_save(state, ckpt_file)

def resume_checkpoint(epoch, model, optimizer, opts):
    print('resume model & optimizer from epoch %d'%(epoch))
    ckpt_file = '%s/ckpt/ep-%04d.pt'%(opts.saveDir, epoch)
    if os.path.isfile(ckpt_file):
        L = torch.load(ckpt_file)
        for key in model.keys():
            _m = model[key]
            if hasattr(_m, 'module'): _m = _m.module
            if 'model_'+key in L: _m.load_state_dict(L['model_'+key])
        for key in optimizer.keys():
            if 'optimizer_'+key in L:
                optimizer[key].load_state_dict(L['optimizer_'+key])
    else:
        print('checkpoint "%s" not found'%(ckpt_file))
        quit()

def initialize(model, initModel):
    if initModel == '':
        print('no further initialization')
        return
    elif os.path.isfile(initModel):
        L = torch.load(initModel)
        for key in model.keys():
            _m = model[key]
            if hasattr(_m, 'module'): _m = _m.module
            if 'model_'+key in L: _m.load_state_dict(L['model_'+key], strict=False)
        print('model initialized using [%s]'%(initModel))
    else:
        print('[%s] not found'%(initModel))
        quit()

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

def train_template(epoch, trLD, model, optimizer):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].train()

    tags = []
    epL = AverageMeters(tags)
    max_itr = max(1, round(len(trLD) * opts.trRatio))
    for itr, samples in progressbar.progressbar(enumerate(islice(trLD, max_itr)), max_value=max_itr):
        # iter, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        # forward: samples => outputs, losses
        ## TODO ##
        # backward
        for key in optimizer.keys(): optimizer[key].zero_grad()
        loss.backward()
        for key in optimizer.keys(): optimizer[key].step()

        # logging
        values = list(map(lambda x: x.item(), []))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values): epL[tag].update(value, btSz)
        # TensorBoard
        if opts.board is not None and itr%opts.dispIter==0:
            for tag, value in zip(tags, values):
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(itr+1)/max_itr)
            # board_vis(epoch, realA, fakeB, realB)
    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

def evaluate_template(epoch, evalLD, model):
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].eval()

    tags = []
    epL = AverageMeters(tags)
    for itr, samples in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # itr, samples = 0, next(iter(evalLD))
        btSz = samples[0].shape[0]
        # forward: samples => outputs, metrics
        ## TODO ##
        # logging
        values = []
        assert len(tags) == len(values)
        for tag, value in zip(tags, values): epL[tag].update(value, btSz)
    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('eval_epoch/'+tag, value, epoch)
