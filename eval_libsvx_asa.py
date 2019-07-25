import h5py
from utils import *

parser = get_empty_parser()
parser.add_argument('--dataset', default='DAVIS16', type=str)
parser.add_argument('--method', default='GBH', type=str)
parser.add_argument('--K', default=1000, type=int)
opts = parser.parse_args()
dataset, method, K = opts.dataset, opts.method, opts.K

libsvxRoot = '/private/home/yangwangx/benchmarks/LIBSVXv4.0/'
ResRoot = os.path.join(libsvxRoot, 'Results', dataset, method, 'Segments')
GtRoot = os.path.join(libsvxRoot, 'Data', dataset, 'GroundTruth')
read_svMap = lambda fname: np.array(h5py.File(fname)['svMap'])
# get all videos
videos = list(os.path.basename(p) for p in glob.glob(GtRoot + '/*'))
NSV, ASA = [], []
for video in progressbar.progressbar(videos):
    resDir = os.path.join(ResRoot, video)
    gtDir = os.path.join(GtRoot, video)
    if True:
        gtFiles = sorted(glob.glob(gtDir + '/*.png'))
        gt = np.asarray(list(imageio.imread(f) for f in gtFiles))  # L H W
    if True:
        resFiles = sorted(glob.glob(resDir + '/*.mat'))
        # np.array(h5py.File(f)['svMap']).min() is always 1
        nSVs = list(read_svMap(f).max() for f in resFiles)
        fIdx = np.argmin(list(abs(nsv-K) for nsv in nSVs))
        svMap = read_svMap(resFiles[fIdx])  # L W H
        svMap = np.moveaxis(svMap, 2, 1)  # L H W
        nsv = nSVs[fIdx]
    if True:
        gtList = gt.flatten().tolist()
        spList = svMap.flatten().tolist()
        asa = computeASA(spList, gtList, 0)
        NSV.append(nsv)
        ASA.append(asa)
listmean = lambda l: sum(l) / len(l)
print('On average: {} ASA @ {} svPerVideo'.format(listmean(ASA), listmean(NSV)))