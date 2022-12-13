import pickle as pkl
import numpy as np

import torch.utils.data as data
from data import common



class SatData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.scale = args.scale if train else args.scale_test

        with open('./dataset/info.pkl', 'rb') as f:
            self.info=common.dotdict(pkl.load(f))
            self.info.root = self.args.dir_data
        self.index_list = common.check_files(self.info)
        if train:
            self.repeat = args.test_every // len(self.index_list)

    def __len__(self):
        if self.train:
            return len(self.index_list) * self.repeat
        else:
            return len(self.index_list)

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        indexs = self.index_list[idx]

        rgb_lr = common.load_raster(self.info, indexs+[-1,])[:,:,:3]
        rgb = common.load_raster(self.info, indexs+[0,])[:,:,:3]
        sentinel = common.load_raster(self.info, indexs+[1,])
        planet = common.load_raster(self.info, indexs+[2,])
        filename = common.idx2name(self.info, indexs+[0,])

        sentinel, planet, rgb_lr, rgb = self.get_patch(sentinel, planet, rgb_lr, rgb)
        sentinel, planet, rgb_lr, rgb = common.np2Tensor(
            sentinel, planet, rgb_lr, rgb, rgb_range=self.args.rgb_range
        )

        return sentinel, planet, rgb_lr, rgb, filename

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.index_list)
        else:
            return idx

    def get_patch(self, lr, hr, lr_rgb, hr_rgb):
        """ Every image has a different aspect ratio. In order to make 
            the input shape the same, here we crop a 96*96 patch on LR 
            image, and crop a corresponding area(96*r, 96*r) on HR image.
        Args:
            args: lr, hr
        Returns:
            0: cropped lr image.
            1: cropped hr image.
        """
        scale = self.scale

        if self.train:
            lr, hr, lr_rgb, hr_rgb = common.get_patch(
                lr,
                hr,
                lr_rgb,
                hr_rgb,
                patch_size=self.args.patch_size,
                scale=scale,
                its=(0, 1, 0)
            )
            if not self.args.no_augment:
                lr, hr, lr_rgb, hr_rgb = common.augment(lr, hr, lr_rgb, hr_rgb)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:int(ih * scale), 0:int(iw * scale)]
            hr_rgb = hr_rgb[0:int(ih * scale), 0:int(iw * scale)]

        return lr, hr, lr_rgb, hr_rgb