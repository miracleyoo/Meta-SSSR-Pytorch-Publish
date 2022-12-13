import os
import json
import torch
import imageio
import numpy as np
from pathlib2 import Path
from data import meta_fm_sr_data as srdata
from data import common


class MULTI(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=False):
        super(MULTI, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(MULTI, self)._set_filesystem(dir_data)
        print(self.dir_hr)
        print(self.dir_lr)

    def _load_file(self, idx):
        """ Load a hr/lr image pair at certain index.
        Args:
            idx: index of data.
        Returns:
            lr: low resolution image as numpy array.
            hr: high resolution image as numpy array.
            filename: filename of the loaded hr image.
        """
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        if self.flag.find('bin') >= 0:
            # TODO: no rgb in multi support
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.flag == 'img' or self.benchmark:
                dataset_name = Path(f_hr).parts[-3]
                folder_name_rgb = dataset_name+'_RGB'
                f_hr_rgb = f_hr.replace(dataset_name, folder_name_rgb)
                f_lr_rgb = f_lr.replace(dataset_name, folder_name_rgb)

                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
                hr_rgb = imageio.imread(f_hr_rgb)
                lr_rgb = imageio.imread(f_lr_rgb)

            elif self.flag.find('sep') >= 0:
                dataset_name = Path(f_hr).parts[-4]
                folder_name_rgb = dataset_name+'_RGB'
                f_hr_rgb = f_hr.replace(dataset_name, folder_name_rgb)
                f_lr_rgb = f_lr.replace(dataset_name, folder_name_rgb)

                with open(f_hr, 'rb') as _f:
                    hr = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_lr, 'rb') as _f:
                    lr = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_hr_rgb, 'rb') as _f:
                    hr_rgb = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_lr_rgb, 'rb') as _f:
                    lr_rgb = np.load(_f, allow_pickle=True)[0]['image']
        return lr, hr, lr_rgb, hr_rgb, filename

    def __getitem__(self, idx):
        # TODO: Remove
        if self.counter % self.args.batch_size == 0 and self.train:
            self._load_wavelength()

        lr, hr, lr_rgb, hr_rgb, filename = self._load_file(idx)

        if hasattr(self, 'wl_in_choice') and len(self.wl_in_choice) != lr.shape[2]:
            lr = np.concatenate([lr[:, :, i][:, :, np.newaxis]
                                 for i in self.wl_in_choice], axis=2)
            if self.args.wl_out_type == 'same':
                hr = np.concatenate([hr[:, :, i][:, :, np.newaxis]
                                     for i in self.wl_in_choice], axis=2)
        if hasattr(self, 'wl_out_choice'):
            hr = np.concatenate([hr[:, :, i][:, :, np.newaxis]
                                    for i in self.wl_out_choice], axis=2) 

        lr, hr, lr_rgb, hr_rgb = self.get_patch(lr, hr, lr_rgb, hr_rgb)

        lr, hr, lr_rgb, hr_rgb = common.set_channel(
            lr, hr, lr_rgb, hr_rgb, n_channels=self.args.n_colors)

        lr_tensor, hr_tensor, lr_rgb_tensor, hr_rgb_tensor = common.np2Tensor(
            lr, hr, lr_rgb, hr_rgb, rgb_range=self.args.rgb_range
        )
        self.counter += 1

        return lr_tensor,\
            hr_tensor,\
            lr_rgb_tensor,\
            hr_rgb_tensor,\
            filename,\
            torch.from_numpy(np.array(self.cwl_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.cwl_out, dtype=np.float32)),\
            torch.from_numpy(np.array(self.bw_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.bw_out, dtype=np.float32)),\
            torch.from_numpy(np.array(self.mean_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.mean_out, dtype=np.float32)),\
            torch.from_numpy(np.array(self.std_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.std_out, dtype=np.float32))

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
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr, lr_rgb, hr_rgb = common.get_patch(
                lr,
                hr,
                lr_rgb,
                hr_rgb,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale,
                its=(0, 1, 0)
            )
            if not self.args.no_augment:
                lr, hr, lr_rgb, hr_rgb = common.augment(lr, hr, lr_rgb, hr_rgb)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:int(ih * scale), 0:int(iw * scale)]
            hr_rgb = hr_rgb[0:int(ih * scale), 0:int(iw * scale)]

        return lr, hr, lr_rgb, hr_rgb
