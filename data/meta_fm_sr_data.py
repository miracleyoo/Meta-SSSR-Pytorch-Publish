import os
import glob

import torch
import json
import pickle
import imageio
import random
import numpy as np

import torch.utils.data as data
import scipy.io as sio
from data import common
# import matplotlib.pyplot as plt
# from PIL import Image


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False, flag=''):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale if train else args.scale_test
        self.idx_scale = 0
        self._test_in_bands_num = 5
        self.counter = 0
        self.flag = args.ext if flag == '' else flag

        self._set_filesystem(args.dir_data)
        self._load_wavelength()

        # Set data index range according to train/test
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        if self.flag.find('img') < 0:
            self.path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(self.path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        # If self.flag contains "bin", then self.images_hr and self.images_lr
        # will be a loaded image numpy array. Otherwise, they are lists of
        # file names of images(ext="img") or .pt(ext="sep") separate bin files.
        if self.flag.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            # list_hr, list_lr = self._scan()
            # hr images list.
            self.images_hr = self._check_and_load(
                self.flag, list_hr, self._name_hrbin()
            )
            # lr images list.
            self.images_lr = [
                self._check_and_load(self.flag, l, self._name_lrbin(s))
                for s, l in zip(self.scale, list_lr)
            ]
        else:
            if self.flag.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif self.flag.find('sep') >= 0:
                # Build a set of folders in 'bin' which have the same structure of the original dataset folder.
                os.makedirs(
                    self.dir_hr.replace(self.apath, self.path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, self.path_bin),
                            'X{:.2f}'.format(s)
                        ),
                        exist_ok=True
                    )

                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                for h in list_hr:
                    b = h.replace(self.apath, self.path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    # print(b, h, self.path_bin, self.apath)
                    self.images_hr.append(b)
                    self._check_and_load(
                        self.flag, [h], b, verbose=True, load=False
                    )

                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, self.path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(
                            self.flag, [l], b, verbose=True, load=False
                        )

        if train:
            self.repeat \
                = args.test_every // (len(self.images_hr) // 16) #args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        """ Scan the target directories and return the hr and lr pathes list
            according to the scale selected.
        """
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )

        if len(names_hr) == 0:
            index_name = os.path.join(self.apath, "index.json")
            if os.path.isfile(index_name):
                with open(index_name, 'r') as f:
                    names_hr = [os.path.join(self.dir_hr, name)
                                for name in json.load(f)]
            else:
                raise FileNotFoundError("Dataset file not found!")

        names_lr = [[] for _ in self.scale]
        print(self.name+" data num:", len(names_hr))
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{:.2f}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        """ Setup some basic dataset settings.
        """
        # root dir of the selected dataset.
        self.apath = os.path.join(dir_data, self.name)
        # hr dir root
        self.dir_hr = os.path.join(self.apath, 'HR')
        # lr dir root
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')

        # Load dataset config
        data_cfg_path = os.path.join(self.apath, self.args.data_config)
        if os.path.isfile(data_cfg_path):
            with open(data_cfg_path, "r") as f:
                self.data_cfg = json.load(f)
        else:
            data_cfg_path = os.path.join(dir_data, self.args.data_config)
            if os.path.isfile(data_cfg_path):
                with open(data_cfg_path, "r") as f:
                    self.data_cfg = json.load(f)
            else:
                raise FileNotFoundError(
                    "No dataset config nor default config file found!")

        # Real file extention for (hr, lr)
        self.ext = (self.data_cfg["hr_suffix"], self.data_cfg["lr_suffix"])

    def _load_wavelength(self):
        """ Load the wavelength and bandwidth information from dataset config file.
        """
        cwl = self.data_cfg["wavelength"]
        bw = self.data_cfg["bandwidth"]
        mean = self.data_cfg["mean"]
        std = self.data_cfg["std"]

        if self.args.wl_in_type != 'all':
            if len(cwl) > self.args.min_in_channel:
                self.in_channel_num = random.randint(self.args.min_in_channel, self.args.max_in_channel)

                if not self.train and self.args.wl_out_type != "comp" and self.args.wl_in_type != 'set':
                    self.wl_in_choice = common.even_sample(range(len(cwl)), self._test_in_bands_num)

                elif self.args.wl_in_type == 'even':
                    if not self.args.jitter:
                        r = range(len(cwl))
                    else:
                        jitter = random.randint(-2, 2)
                        if jitter >= 0:
                            r = range(jitter, len(cwl))
                        else:
                            r = range(0, len(cwl)+jitter)
                        
                    self.wl_in_choice = common.even_sample(r, self.in_channel_num)
                elif self.args.wl_in_type == 'rand':
                    self.wl_in_choice = random.sample(range(len(cwl)), self.in_channel_num)
                elif self.args.wl_in_type == 'mid2side':
                    start = (len(cwl) - self.args.in_bands_num)//2 
                    end = start + self.args.in_bands_num
                    self.wl_in_choice = list(range(start, end))
                elif self.args.wl_in_type == 'start2end':
                    self.wl_in_choice = list(range(self.args.in_bands_num))
                elif self.args.wl_in_type == 'set':
                    self.wl_in_choice = self.args.bands_in_set


                self.cwl_in = [cwl[i] for i in self.wl_in_choice]
                self.bw_in = [cwl[i] for i in self.wl_in_choice]
                self.mean_in = [mean[i] for i in self.wl_in_choice]
                self.std_in = [std[i] for i in self.wl_in_choice]
        else:
            self.cwl_in = cwl
            self.bw_in = bw
            self.mean_in = mean
            self.std_in = std
            self.in_channel_num = len(self.cwl_in)


        if self.args.wl_out_type == "rgb":
            self.cwl_out = self.args.cwl_rgb
            self.bw_out = [self.args.rgb_avg_bw]*3
            self.mean_out = self.args.mean_rgb
            self.std_out = self.args.std_rgb
        elif self.args.wl_out_type == "all":
            self.cwl_out = self.args.cwl_hsi
            self.bw_out = [self.args.hsi_avg_bw]*31
            self.mean_out = self.args.mean_hsi
            self.std_out = self.args.std_hsi
        elif self.args.wl_out_type == "max":
            self.cwl_out = cwl
            self.bw_out = bw
            self.mean_out = mean
            self.std_out = std
        elif self.args.wl_out_type == "set":
            self.cwl_out = self.args.cwl_set
            self.bw_out = self.args.avg_bw_set
            self.mean_out = self.args.mean_set
            self.std_out = self.args.std_set
        elif self.args.wl_out_type == "same":
            self.cwl_out = self.cwl_in
            self.bw_out = self.bw_in
            self.mean_out = self.mean_in
            self.std_out = self.std_in
        elif self.args.wl_out_type == "rand":
            self.out_channel_num = min(2*self.in_channel_num, len(cwl))
            self.wl_out_choice = random.sample(
                range(len(cwl)), self.out_channel_num)
            self.cwl_out = [cwl[i] for i in self.wl_out_choice]
            self.bw_out = [bw[i] for i in self.wl_out_choice]
            self.mean_out = [mean[i] for i in self.wl_out_choice]
            self.std_out = [std[i] for i in self.wl_out_choice]
        elif self.args.wl_out_type == "comp":
            self.wl_out_choice = [i for i in range(len(cwl)) if i not in self.wl_in_choice]
            self.cwl_out = [cwl[i] for i in self.wl_out_choice]
            self.bw_out = [bw[i] for i in self.wl_out_choice]
            self.mean_out = [mean[i] for i in self.wl_out_choice]
            self.std_out = [std[i] for i in self.wl_out_choice]
        else:
            raise KeyError(
                "Wavelength output type should only be inside rgb|all|set|same.")

    def _name_hrbin(self):
        """ Return the name of compressed bin file of the hr
            images(a list of names and images loaded).
        """
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        """ Return the name of compressed bin file of the hr
            images(a list of names and images loaded).
        """
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        """ Load the whole dataset and compress it into a bin
            file if not existing.
        Args:
            ext: Load strategy flag.
            l: Pathes of images being loaded.
            f: Saving bin file name.
        """
        # Load the existing file f.
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose:
                    print('Loading {}...'.format(f))
                with open(f, 'rb') as _f:
                    ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f:
                pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        # TODO: Remove
        if self.counter % self.args.batch_size == 0 and self.train:
            self._load_wavelength()
            # print(self.wl_in_choice)
            
        lr, hr, filename = self._load_file(idx)

        if hasattr(self, 'wl_in_choice') and len(self.wl_in_choice) != lr.shape[2]:
            lr = np.concatenate([lr[:, :, i][:, :, np.newaxis]
                                 for i in self.wl_in_choice], axis=2)
            if self.args.wl_out_type == 'same':
                hr = np.concatenate([hr[:, :, i][:, :, np.newaxis]
                                     for i in self.wl_in_choice], axis=2)
        if hasattr(self, 'wl_out_choice'):
            hr = np.concatenate([hr[:, :, i][:, :, np.newaxis]
                                    for i in self.wl_out_choice], axis=2) 

        lr, hr = self.get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )
        self.counter += 1

        return lr_tensor,\
            hr_tensor,\
            filename,\
            torch.from_numpy(np.array(self.cwl_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.cwl_out, dtype=np.float32)),\
            torch.from_numpy(np.array(self.bw_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.bw_out, dtype=np.float32)),\
            torch.from_numpy(np.array(self.mean_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.mean_out, dtype=np.float32)),\
            torch.from_numpy(np.array(self.std_in, dtype=np.float32)),\
            torch.from_numpy(np.array(self.std_out, dtype=np.float32))

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

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
        print(f_hr)

        if self.flag.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
            
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            print(filename)
            if self.flag == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.flag.find('sep') >= 0:
                with open(f_hr, 'rb') as _f:
                    hr = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_lr, 'rb') as _f:
                    lr = np.load(_f, allow_pickle=True)[0]['image']

        return lr, hr, filename

    def get_patch(self, lr, hr):
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
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:int(ih * scale), 0:int(iw * scale)]

        return lr, hr

    def set_scale(self, idx_scale):
        """ Set the scale which is used currently.
        """
        self.idx_scale = idx_scale

    @property
    def test_in_bands_num(self):
        return self._test_in_bands_num

    @test_in_bands_num.setter
    def test_in_bands_num(self, value):
        self._test_in_bands_num = value
        self._load_wavelength()
