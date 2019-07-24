import numpy as np
from scipy.ndimage import zoom
from scipy import interpolate
from skimage.segmentation import mark_boundaries

def get_rand_scale_factor():
    """draws a value from Normal(1, 0.75), clips it between [0.75, 3.0]"""
    return np.max((0.75, np.min((3.0, np.random.normal(1, 0.75)))))

def scale_image(img, s_factor):
    """scales image (numpy array [H, W, 3]) by s_factor"""
    s_img = zoom(img, (s_factor, s_factor, 1), order=1)
    return s_img

def scale_label(label, s_factor):
    """scales segmentation label (numpy array [H, W]) by s_factor"""
    s_label = zoom(label, (s_factor, s_factor), order = 0)
    return s_label

def get_spixel_image(img, spix_index):
    """marks superpixel boundaries on the image"""
    spixel_image = mark_boundaries(img / 255., spix_index.astype(int), color=(1, 1, 0))
    spixel_image = (spixel_image * 255).astype(np.uint8)
    return spixel_image

def get_spixel_init(num_spixels, img_height, img_width):
    """computes initial superpixel index maps"""
    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    spixel_height = img_height / k_h
    spixel_width = img_width / k_w

    h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1, spixel_height)
    w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1, spixel_width)
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height, img_width))
    spixel_initmap = spixel_initmap[None, :, :]  # 1 H W
    return spixel_initmap, k_h, k_w

def convert_label(label, max_segments=50):
    """converts segmentation label to one-hot vector.
    Args:
        label: numpy array, [H, W]
    Returns:
        label2: numpy array, [1, H, W]
        problabel: numpy array, [50, H, W]
    """
    H, W = label.shape
    problabel = np.zeros((max_segments, H, W), dtype=np.float32)
    label_list = np.unique(label).tolist()
    for i, lb in enumerate(label_list):
        if i < max_segments:
            problabel[i] = (label == lb)
        else:
            # print('\nOnly {} out of {} segments are used'.format(max_segments, len(label_list)))
            break    
    label2 = np.squeeze(np.argmax(problabel, axis=0))  # H W
    label2 = np.expand_dims(label2, axis=0)            # 1 H W
    return label2, problabel
