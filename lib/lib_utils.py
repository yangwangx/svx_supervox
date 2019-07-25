import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF


from skimage.segmentation import mark_boundaries
def get_spixel_image(img, spix_index):
    """marks superpixel boundaries on the image"""
    spixel_image = mark_boundaries(img / 255., spix_index.astype(int), color=(1, 0, 0))
    spixel_image = (spixel_image * 255).astype(np.uint8)
    return spixel_image

from scipy import interpolate
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
    spixel_initmap = spixel_initmap  # H W
    return spixel_initmap, k_h, k_w