import random

import numpy as np
import skimage.color as sc
import rasterio as rio
import pickle as pkl

from operator import itemgetter
from pathlib2 import Path

import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_patch(*args, patch_size=96, scale=1, its=None):
    """ Every image has a different aspect ratio. In order to make 
        the input shape the same, here we crop a 96*96 patch on LR 
        image, and crop a corresponding area(96*r, 96*r) on HR image.
    Args:
        args: lr, hr
        patch_size: The x and y length of the crop area on lr image.
        scale: r, upscale ratio
    Returns:
        0: cropped lr image.
        1: cropped hr image.
    """
    ih, iw = args[0].shape[:2]

    tp = int(scale * patch_size)
    ip = patch_size

    ix = random.randrange(0, (iw-ip))
    iy = random.randrange(0, (ih-ip))

    tx, ty = int(scale * ix), int(scale * iy)

    if its is None:
        its = np.zeros(len(args)-1, int)
    itp = (tp, ip)
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[(ty, iy)[b]:(ty, iy)[b] + itp[b], (tx, ix)[b]:(tx, ix)[b] + itp[b], :] for a, b in zip(args[1:], its)]
    ]
    return ret


def set_channel(*args, n_channels=3):
    """ Do the channel number check. If input channel is 
        not n_channels, convert it to n_channels.
    Args:
        n_channels: the target channel number.
    """
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    """ Transform an numpy array to tensor. Each single value in
        the target tensor will be mapped into [0,1]
    Args:
        rgb_range: Max value of a single pixel in the original array.
    """
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

def even_sample(items, number):
    indexs = (np.linspace(0, len(items)-1, number).astype(int)).tolist()
    return itemgetter(*indexs)(items)


def load_raster(info, indexs, use_pkl=True):
    if indexs[2] == -1:
        indexs[2] = 0
        suf = '_lr'
    else:
        suf = ''
    try:
        place = info.places[indexs[0]]
        date  = info.dates[indexs[1]]
        stype = info.stypes[indexs[2]]
    except IndexError:
        return None
    data_name = f'{date}_{stype}_{info.suffix}' if info.suffix is not None else f'{date}_{stype}'
    data_root = Path(info.root) / place / data_name
    if data_root.exists():
        if use_pkl:
            data_path = str(data_root / (data_name+suf+'.pkl'))
            with open(data_path, 'rb') as f:
                pkl_img = pkl.load(f)
            return pkl_img
        else:
            data_path = str(data_root / (data_name+'.tif'))
            return rio.open(data_path).read().transpose(1,2,0)
    else:
        return None

def idx2name(info, indexs):
    try:
        place = info.places[indexs[0]]
        date  = info.dates[indexs[1]]
        stype = info.stypes[indexs[2]]
    except IndexError:
        return None

    filename = f'{place}_{date}_{stype}_{info.suffix}' if info.suffix is not None else f'{date}_{stype}'
    return filename

def check_files(info):
    valid_idxs = []
    for i in range(len(info.places)):
        for j in range(len(info.dates)):
            place = info.places[i]
            date  = info.dates[j]
            stype = info.stypes[0]
            data_name = f'{date}_{stype}_{info.suffix}' if info.suffix is not None else f'{date}_{stype}'
            data_root = Path(info.root) / place / data_name
            if data_root.exists():
                valid_idxs.append([i,j])
    return valid_idxs

