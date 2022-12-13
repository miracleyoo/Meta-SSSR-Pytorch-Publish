import multiprocessing as mp
import pickle as pkl
from functools import partial
from glob import glob

import numpy as np
from pathlib2 import Path

import cv2

from mlib import pf


def rescale_x1_x4(path, out_root):
    print("==> Start:", path)
    with open(path, 'rb') as f:
        hsi = pkl.load(f)[0]['image']
    path = Path(path)
    out_root = pf.get_folder(out_root)
    for scale in np.linspace(1.1, 4.0, 30):
        lr = cv2.resize(hsi, tuple(map(lambda x: int(x/scale),
                                       hsi.shape[:-1]))[::-1], interpolation=cv2.INTER_CUBIC)
        out_path = pf.get_folder(
            out_root / "X{0:.2f}".format(scale)) / path.name
        with open(str(out_path), 'wb') as _f:
            pkl.dump([{'name': path.stem, 'image': lr}], _f)


if __name__ == "__main__":
    roots = [r"/mnt/d/Dataset/Multispectral/ICVL_HSI",
             r"/mnt/d/Dataset/Multispectral/NTIRE2020/NTIRE2020_train",
             r"/mnt/d/Dataset/Multispectral/NTIRE2020/NTIRE2020_val",
             ]
    for root in roots:
        root = Path(root)
        mat_root = root / "mat"
        rgb_root = root / "rgb"
        rgb_bin_root = root / "rgb_bin"
        hsi_bin_root = root / "hsi_bin"

        hsi_bin_files = glob(str(hsi_bin_root/"*.pt"))

        with mp.Pool() as pool:
            print("Start pooling!")
            results = pool.map(partial(rescale_x1_x4, out_root=(
                root/"HSI_LR_bicubic")), hsi_bin_files)

        rgb_bin_files = glob(str(rgb_bin_root/"*.pt"))
        with mp.Pool() as pool:
            print("Start pooling!")
            results = pool.map(partial(rescale_x1_x4, out_root=(
                root/"RGB_LR_bicubic")), rgb_bin_files)
