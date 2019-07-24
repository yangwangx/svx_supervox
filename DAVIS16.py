import os, sys, glob, random, numpy as np
import imageio
from skimage.util import img_as_uint, img_as_float
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import zoom
import torch
import torch.utils.data as DD
try:
    cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
except NameError:
    cwd = ''

__all__ = ['DAVIS16_train', 'DAVIS16_eval']

dataRoot = os.path.join(cwd, 'davis2016')
trainRoot = os.path.join(dataRoot, 'DAVIS/JPEGImages/480p/')  # 00000.jpg
trainGTRoot = os.path.join(dataRoot, 'DAVIS/Annotations/480p/')  # 00000.png
evalRoot = os.path.join(dataRoot, 'libsvx/PNGImages/')   # 00001.png
evalGTRoot = os.path.join(dataRoot, 'libsvx/GroundTruth/')  # 00001.png

def get_trainDirs():
    trainSplitFile = os.path.join(dataRoot, 'DAVIS/split/train.txt')
    trainVideoDirs, trainGTDirs = [], []
    with open(trainSplitFile, 'r') as f:
        for line in f.readlines():
            video = line.rstrip()
            trainVideoDirs.append(os.path.join(trainRoot, video))
            trainGTDirs.append(os.path.join(trainGTRoot, video))
    return trainVideoDirs, trainGTDirs

def get_evalDirs():
    evalVideoDirs = sorted(glob.glob(os.path.join(evalRoot, '*')))
    evalGTDirs = sorted(glob.glob(os.path.join(evalGTRoot, '*')))
    return evalVideoDirs, evalGTDirs

def count_file(dir, ext):
    return len(glob.glob(os.path.join(dir, '*.'+ext)))

def convert_binary_gt(gts):
    L, H, W = gts.shape
    onehots = np.zeros((2, L, H, W), dtype=np.float32)  # 2 L H W
    onehots[0] = (gts == 0)
    onehots[1] = (gts == 1)
    return onehots

def get_rand_scale_factor():
    """draws a value from Normal(1, 0.5), clips it between [0.5, 1.5]"""
    return np.max((0.5, np.min((1.5, np.random.normal(1, 0.5)))))

def scale_image(img, s_factor):
    """scales image (numpy array [H, W, 3]) by s_factor"""
    s_img = zoom(img, (s_factor, s_factor, 1), order=1)
    return s_img

def scale_label(label, s_factor):
    """scales segmentation label (numpy array [H, W]) by s_factor"""
    s_label = zoom(label, (s_factor, s_factor), order=0)
    return s_label

def fix_label_shape(label):
    # `GroundTruth/bear/00077.png` has four channels
    return label if len(label.shape) == 2 else label[:, :, 0]

class DAVIS16_eval(DD.Dataset):
    def __init__(self):
        super(DAVIS16_eval, self).__init__()
        self.VideoDirs, self.GTDirs = get_evalDirs()

    def __getitem__(self, index):
        videoDir = self.VideoDirs[index]
        gtDir = self.GTDirs[index]
        nFrame = count_file(gtDir, ext='png')
        imFiles = list('{}/{:05d}.png'.format(videoDir, t+1) for t in range(nFrame))
        gtFiles= list('{}/{:05d}.png'.format(gtDir, t+1) for t in range(nFrame))
        # read frames
        ims = np.asarray(list(rgb2lab(img_as_float(imageio.imread(imFile))) for imFile in imFiles))  # L H W 3
        # read gtsegs
        gts = np.asarray(list(imageio.imread(gtFile) for gtFile in gtFiles))
        gts = (gts >= 1).astype(np.uint8)  # L H W
        # numpy to tensor
        ims = torch.Tensor(np.moveaxis(ims, 3, 0))  # 3 L H W
        gts = torch.ByteTensor(gts)  # L H W
        return ims, gts

    def __len__(self):
        return len(self.VideoDirs)

class DAVIS16_train(DD.Dataset):
    def __init__(self, crop_size=[16, 201, 201]):
        super(DAVIS16_train, self).__init__()
        assert(len(crop_size) == 3)
        self.VideoDirs, self.GTDirs = get_trainDirs()
        self.crop_size = crop_size

    def __getitem__(self, index):
        videoDir = self.VideoDirs[index % len(self.VideoDirs)]
        gtDir = self.GTDirs[index % len(self.GTDirs)]
        nFrame = count_file(gtDir, ext='png')
        cL, cH, cW = self.crop_size
        # temporal crop & flip
        t0 = random.randrange(0, nFrame-cL+1)  # [0, cL), [nFrame-cL, nFrame)
        ts = list(range(t0, t0+cL))
        if random.random() > 0.5:
            ts = ts[::-1]
        imFiles = list('{}/{:05d}.jpg'.format(videoDir, t) for t in ts)
        gtFiles= list('{}/{:05d}.png'.format(gtDir, t) for t in ts)
        # spatial crop
        s_factor = get_rand_scale_factor()
        H, W, _ = scale_image(img_as_float(imageio.imread(imFiles[0])), s_factor).shape
        y0 = random.randrange(0, H-cH+1)
        x0 = random.randrange(0, W-cW+1)
        # read images
        ims = np.asarray(list(
            rgb2lab(scale_image(img_as_float(imageio.imread(imFile)), s_factor)[y0:y0+cH, x0:x0+cW, :])
            for imFile in imFiles))
        gts = np.asarray(list(
            scale_label(fix_label_shape(imageio.imread(gtFile)), s_factor)[y0:y0+cH, x0:x0+cW]
            for gtFile in gtFiles))
        gts = (gts >= 1).astype(np.uint8)
        # spatial flip
        if random.random() > 0.5:
            ims = ims[:, :, ::-1, :]  # L H W 3
            gts = gts[:, :, ::-1]  # L H W
        onehots = convert_binary_gt(gts)  # 2 L H W
        # numpy to tensor
        ims = torch.Tensor(np.moveaxis(ims, 3, 0).copy())  # 3 L H W
        gts = torch.ByteTensor(gts.copy())  # L H W
        onehots = torch.Tensor(onehots)  # 2 L H W
        return ims, gts, onehots

    def __len__(self):
        return len(self.VideoDirs) * 4  # repeat 4 times

if __name__ == '__main__':
    visual_check = True
    if True:
        dataset = DAVIS16_train()
        samples = random.choice(dataset)
        for x in samples: print(x.dtype, x.shape)
        if visual_check:
            ims, gts, onehots = samples
            im = img_as_uint(lab2rgb(np.moveaxis(ims[:, 5].numpy(), 0, 2))) # H W 3, uint8
            gt = img_as_uint(np.repeat(gts[5].numpy()[:, :, None], 3, axis=2).astype(np.float32)) # H W 3, uint8
            imageio.imwrite('temp1.png', np.concatenate([im, gt], axis=1))
    if True:
        dataset = DAVIS16_eval()
        samples = random.choice(dataset)
        for x in samples: print(x.dtype, x.shape)
        if visual_check:
            ims, gts = samples
            im = img_as_uint(lab2rgb(np.moveaxis(ims[:, 5].numpy(), 0, 2))) # H W 3, uint8
            gt = img_as_uint(np.repeat(gts[5].numpy()[:, :, None], 3, axis=2).astype(np.float32)) # H W 3, uint8
            imageio.imwrite('temp2.png', np.concatenate([im, gt], axis=1))