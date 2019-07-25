from DAVIS16 import *
from utils import *
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

parser = get_base_parser()
# data
parser.add_argument('--crop_size', default=[16, 201, 201], type=int, nargs='+', dest='crop_size')
# model input
parser.add_argument('--p_scale', default=0.25, type=float, help='control factor for TYX channel')
parser.add_argument('--lab_scale', default=0.26, type=float, help='scale factor for LAB channel')
parser.add_argument('--n_sv', default=100, type=int, help='number of superpixels in frame')
parser.add_argument('--t_sv', default=3, type=int, help='number of superpixels in time')
# cnn
parser.add_argument('--no_cnn', dest='use_cnn', default=True, action='store_false')
# kmeans
parser.add_argument('--unfold', default=5, type=int)
parser.add_argument('--softscale', default=-1.0, type=float)
parser.add_argument('--hier_ratio', default=1.0, type=float)
# loss
parser.add_argument('--w_pos', default=0.1, type=float)
parser.add_argument('--w_col', default=0.0, type=float)
parser.add_argument('--w_label', default=10, type=float)
# mode
parser.add_argument('--nSliceEval', default=1, type=int, help='number of slices for evaluation')
opts = parser.parse_args()
opts.BtCount = 0
manual_seed(opts.seed)

def create_dataloader():
    trSet = DAVIS16_train(crop_size=opts.crop_size)
    trLD = DD.DataLoader(trSet, batch_size=opts.BtSz,
        sampler= DD.sampler.SubsetRandomSampler([0]*opts.BtSz) if opts.OneBatch else DD.sampler.RandomSampler(trSet),
        num_workers=opts.workers, pin_memory=True, drop_last=True)
    evalSet = DAVIS16_eval()
    evalLD = DD.DataLoader(evalSet, batch_size=1,
        sampler=DD.sampler.SequentialSampler(evalSet),
        num_workers=opts.workers, pin_memory=True, drop_last=False)
    return trLD, evalLD

def create_model():
    model = edict()
    model.svx = SVX(use_cnn=opts.use_cnn, num_in=6, num_out=14, num_ch=32)
    for key in model.keys():
        model[key] = model[key].to(DEVICE)
        if DEVICE != "cpu": model[key] = nn.DataParallel(model[key])
    return model

def __configure(svx, vid_shape,
               t_sv=opts.t_sv, n_sv=opts.n_sv,
               p_scale=opts.p_scale, lab_scale=opts.lab_scale,
               softscale=opts.softscale, num_steps=opts.unfold):
    if hasattr(svx, 'module'):
        svx = svx.module
    return svx.configure(vid_shape, t_sv, n_sv, p_scale, lab_scale, softscale, num_steps)

def __enforce_connectivity(svMap):
    segment_size = svMap.size / np.unique(svMap).size
    min_size = int(0.06 * segment_size)
    max_size = int(5 * segment_size)
    return enforce_connectivity(svMap, min_size, max_size)

def get_init_spIndx(vid_shape, t_sv, n_sv):
    B, _, L, H, W = vid_shape
    init_spIndx, Kh, Kw = get_spixel_init(n_sv, H, W)
    init_spIndx = torch.from_numpy(init_spIndx).float() # H W
    init_spIndx = init_spIndx.view(1, H, W).repeat(L, 1, 1)  # L H W
    for t in range(L):
        init_spIndx[t] += Kh * Kw * np.floor(t * t_sv / L)
    init_spIndx = init_spIndx.view(1, 1, L, H, W).expand(B, 1, L, H, W)  # B 1 L H W
    return init_spIndx.to(DEVICE)

def train(epoch, trLD, model, optimizer):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].train()

    tags = ['loss_label', 'loss_pos', 'loss_color', 'loss']
    epL = AverageMeters(tags)
    max_itr = max(1, round(len(trLD) * opts.trRatio))
    for itr, samples in progressbar.progressbar(enumerate(islice(trLD, max_itr)), max_value=max_itr):
        # iter, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        # forward: samples => outputs, losses
        if True:
            vid, label, onehot = list(map(lambda x: x.to(DEVICE), samples))
            L, H, W, Kl, Kh, Kw, Khw, K = __configure(model.svx, vid.shape)
            init_spIndx = get_init_spIndx(vid.shape, t_sv=opts.t_sv, n_sv=opts.n_sv)
            pFeat, spFeat, final_assoc, final_spIndx = model.svx(vid, init_spIndx)
        if True:
            pFeat_tyxlab = pFeat[:, :6].contiguous()
            loss_pos, loss_col, loss_label = \
                compute_svx_loss(pFeat_tyxlab, final_assoc, init_spIndx, final_spIndx, onehot, Kl, Kh, Kw)
            loss = opts.w_label * loss_label + opts.w_pos * loss_pos + opts.w_col * loss_col
        # backward
        if opts.BtCount % opts.BtMerge == 0:
            for key in optimizer.keys(): optimizer[key].zero_grad()
        loss.backward()
        opts.BtCount += 1
        if opts.BtCount % opts.BtMerge == 0:
            for key in optimizer.keys(): optimizer[key].step()
        # logging
        values = list(map(lambda x: x.item(), [loss_label, loss_pos, loss_col, loss]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values): epL[tag].update(value, btSz)
        # TensorBoard
        if opts.board is not None and itr%opts.dispIter==0:
            for tag, value in zip(tags, values):
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(itr+1)/max_itr)
    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

def evaluate(epoch, evalLD, model):
    if opts.nSliceEval > 1:
        assert opts.t_sv >= opts.nSliceEval, 'cannot slice video when nSliceEval > t_sv'
        return evaluate_slice(epoch, evalLD, model)
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].eval()

    tags = ['ASA', 'NSV']
    epL = AverageMeters(tags)
    for itr, samples in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # itr, samples = 0, next(iter(evalLD))
        btSz = samples[0].shape[0]
        # forward: samples => outputs, metrics
        if True:
            with torch.no_grad():
                vid = samples[0].to(DEVICE)
                gtList = samples[1].cpu().long().numpy().flatten().tolist()
                L, H, W, Kl, Kh, Kw, Khw, K = __configure(model.svx, vid.shape)
                init_spIndx = get_init_spIndx(vid.shape, t_sv=opts.t_sv, n_sv=opts.n_sv)
                _, _, _, final_spIndx = model.svx(vid, init_spIndx)
                svMap = np.squeeze(final_spIndx.cpu().numpy(), axis=(0,1)).astype(int)  # L H W
                svMap =  __enforce_connectivity(svMap)
            nSV = svMap.max() + 1
            spList = svMap.flatten().tolist()
            asa = computeASA(spList, gtList, 0)
        # logging
        values = [asa, nSV]
        assert len(tags) == len(values)
        for tag, value in zip(tags, values): epL[tag].update(value, btSz)
    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('eval_epoch/'+tag, value, epoch)

"""When the video is too long, we can slice the video into multiple segments"""
def evaluate_slice(epoch, evalLD, model):
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].eval()

    tags = ['ASA', 'NSV']
    epL = AverageMeters(tags)
    for itr, samples in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # itr, samples = 0, next(iter(evalLD))
        btSz = samples[0].shape[0]
        # forward: samples => outputs, metrics
        if True:
            with torch.no_grad():
                vid = samples[0].to(DEVICE)
                L = vid.shape[2]
                gtList = samples[1].cpu().long().numpy().flatten().tolist()
                # slice
                slice_sv_starts = list(int(np.floor(i*opts.t_sv/opts.nSliceEval)) for i in range(opts.nSliceEval))
                slice_sv_ends = slice_sv_starts[1:] + [opts.t_sv]
                slice_frm_starts = list(int(np.floor(i * L / opts.t_sv)) for i in slice_sv_starts)
                slice_frm_ends   = slice_frm_starts[1:] + [L,]
                # compute sliced svMap
                svMap = []
                for sv_s, sv_e, frm_s, frm_e in zip(slice_sv_starts, slice_sv_ends, slice_frm_starts, slice_frm_ends):
                    _vid = vid[:, :, frm_s:frm_e]  # _vid of this slice
                    _t_sv = sv_e - sv_s  # _t_sv of this slice                    
                    _L, _H, _W, _Kl, _Kh, _Kw, _Khw, _K = __configure(model.svx, _vid.shape, t_sv=_t_sv)
                    _init_spIndx = get_init_spIndx(_vid.shape, t_sv=_t_sv, n_sv=opts.n_sv)
                    _, _, _, _final_spIndx = model.svx(_vid, _init_spIndx)
                    _svMap = np.squeeze(_final_spIndx.cpu().numpy(), axis=(0,1)).astype(int)  # L H W
                    svMap.append(_svMap)
                # merge sliced svMap
                for i in range(1, len(svMap)):
                    svMap[i] += svMap[i-1].max() + 1
                svMap = np.concatenate(svMap, axis=0)
                svMap = __enforce_connectivity(svMap)
            nSV = svMap.max() + 1
            spList = svMap.flatten().tolist()
            asa = computeASA(spList, gtList, 0)
        # logging
        values = [asa, nSV]
        assert len(tags) == len(values)
        for tag, value in zip(tags, values): epL[tag].update(value, btSz)
    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('eval_epoch/'+tag, value, epoch)

def main_train():
    print('==> create dataset loader')
    trLD, evalLD = create_dataloader()

    print('==> create model, optimizer, scheduler')
    model = create_model()
    optimizer = create_optimizer(model, opts)
    scheduler = create_scheduler(optimizer, opts)

    print('==> initialize with checkpoint or initModel ?')
    FIRST_EPOCH = 1 # do not change
    USE_CKPT = opts.checkpoint >= FIRST_EPOCH
    if USE_CKPT:
        resume_checkpoint(opts.checkpoint, model, optimizer, opts)
        start_epoch = opts.checkpoint + 1
    else:
        initialize(model, opts.initModel)
        start_epoch = FIRST_EPOCH

    print('==> start training from epoch %d'%(start_epoch))
    for epoch in range(start_epoch, FIRST_EPOCH + opts.nEpoch):
        print('\nEpoch {}:\n'.format(epoch))
        for key in scheduler.keys():
            scheduler[key].step(epoch-1)
            lr = scheduler[key].optimizer.param_groups[0]['lr']
            print('learning rate of {} is set to {}'.format(key, lr))
            if opts.board is not None: opts.board.add_scalar('lr_schedule/'+key, lr, epoch)
        train(epoch, trLD, model, optimizer)
        if not opts.OneBatch and epoch%opts.saveStep==0:
            save_checkpoint(epoch, model, optimizer, opts)
        if not opts.OneBatch and epoch%opts.evalStep==0:
            evaluate(epoch, evalLD, model)

def main_eval():
    _, evalLD = create_dataloader()
    model = create_model()
    initialize(model, opts.initModel)
    evaluate(-1, evalLD, model)

def main_vis():
    raise NotImplementedError

def main_apply():
    raise NotImplementedError

if __name__ == '__main__':
    trainMode = True
    if opts.evalMode:
        opts.board = None
        trainMode = False
        main_eval()
    if opts.visMode:
        opts.board = None
        trainMode = False
        main_vis()
    if opts.applyMode:
        opts.board = None
        trainMode = False
        main_apply()
    if trainMode:
        opts.board = SummaryWriter(os.path.join(opts.saveDir, 'board'))
        options_text = print_options(opts, parser)
        opts.board.add_text('options', options_text, opts.checkpoint)
        main_train()
        opts.board.close()