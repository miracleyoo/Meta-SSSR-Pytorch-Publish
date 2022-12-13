
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common
from model.matrix import *


def make_model(args, parent=False):
    return MetaRDN(args)


## ------------------ Meta Magnitude Modules -------------------- ##
class WL2Weight(nn.Module):
    """ Generate 1x1 Custom Conv layer weights from wavelength.
    """
    def __init__(self, inC, outC, kernel_size=1, use_bw=True, bias=True):
        super(WL2Weight, self).__init__()
        self.kernel_size = kernel_size
        self.inC = inC
        self.outC = outC
        self.channel = 1+use_bw
        self.use_bw = use_bw
        self.bias = bias

        self.meta_block = nn.Sequential(
            nn.Linear(self.channel, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.outC*(self.inC*self.kernel_size *
                                      self.kernel_size+self.bias)),
            nn.Sigmoid()
        )

    def forward(self, wl, bw):
        """
        Args:
            wl: Central wavelength.
        """
        wl = (wl.float() - 550) / 10
        bw = bw.float() - 55

        if self.use_bw:
            x = torch.stack([wl, bw]).unsqueeze(0)
        else:
            x = wl.unsqueeze(0).unsqueeze(0)

        x = self.meta_block(x)
        if self.bias:
            bias = x[:, -self.outC:].squeeze(0)
            out = x[:, :-self.outC].contiguous().view(self.outC, self.inC,
                                                      self.kernel_size, self.kernel_size)
            return out, bias
        else:
            out = x.contiguous().view(self.outC, self.inC, self.kernel_size, self.kernel_size)
            return out


## ------------------ Full Network ------------------------- ##
class MetaRDN(nn.Module):
    def __init__(self, args):
        super(MetaRDN, self).__init__()
        self.args = args
        self.scale = self.args.scale
        H0 = self.args.H0
        G0 = self.args.G0
        kSize = self.args.RDNkSize
        self.scale_idx = 0
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')

        self.sub_mean = partial(common.mean_shift_1d, add=False)
        self.add_mean = partial(common.mean_shift_1d, add=True)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            1, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(
            G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Central Wavelength weight prediction Network
        self.wlwp1 = WL2Weight(inC=G0, outC=G0,
                               kernel_size=self.args.WLkSize1, use_bw=1-self.args.skip_bw, bias=True)
        self.wlwp2 = WL2Weight(inC=G0+3, outC=G0,
                               kernel_size=self.args.WLkSize2, use_bw=1-self.args.skip_bw, bias=True)

        # Redidual Dense Blocks
        if self.args.head_blocks > 0:
            self.RDBs_head = nn.ModuleList()
            for i in range(self.args.head_blocks):
                self.RDBs_head.append(
                    common.RDB(growRate0=G0, growRate=self.args.G_hidden,
                               nConvLayers=self.args.rdb_conv_num)
                )

            # Global Feature Fusion
            self.GFF_head = nn.Sequential(*[
                nn.Conv2d(self.args.head_blocks * G0,
                          G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
            ])

        m_head = [
            common.ResBlock(
                common.default_conv, G0, 3, act=nn.ReLU(), res_scale=args.res_scale
            ) for _ in range(self.args.head_blocks)
        ]
        self.head = nn.Sequential(*m_head)

        if self.args.body_blocks > 0:
            self.RDBs_body = nn.ModuleList()
            for _ in range(self.args.body_blocks):
                self.RDBs_body.append(
                    common.RDB(growRate0=G0, growRate=self.args.G_hidden,
                               nConvLayers=self.args.rdb_conv_num)
                )

            if (self.args.body_blocks % 2) == 1:
                self.RDBs_body.append(
                    common.RDB(growRate0=G0, growRate=self.args.G_hidden,
                               nConvLayers=self.args.rdb_conv_num)
                )
            self.GFF_body = nn.Sequential(*[
                nn.Conv2d((self.args.body_blocks) *
                          G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
            ])

        m_tail = [
            common.ResBlock(
                common.default_conv, G0, 3, act=nn.ReLU(), res_scale=args.res_scale
            ) for _ in range(self.args.tail_blocks)
        ]
        m_tail.append(common.default_conv(G0, 1, 3))
        self.tail = nn.Sequential(*m_tail)

        self.upsample = nn.Sequential(*[
            common.Upsampler(common.default_conv,
                             self.scale, G0, act=False),
            common.default_conv(G0, G0, 3)])

    def forward(self, x, lr_rgb, hr_rgb, info):
        """ Network forward function.
        Args:
            x: input image tensor.
            pos_mat: position matrix for meta magnitude.
            in_cwl: input central wavelengthes.
            cwl_in: input HSI central wavelengthes.
            cwl_out: output HSI central wavelengthes.
            bw_in: input HSI bandwidthes.
            bw_out: output HSI bandwidthes.
        """
        self.B, self.C, *_ = x.shape

        # ------------------------------------- Head ------------------------------------- #
        head_feat = []
        for i in range(self.C+3):
            if i < self.C:
                slice_x = x[:, i, :, :]
            else:
                slice_x = lr_rgb[:, i-self.C, :, :]

            mean_i = info.mean_in[i].repeat(self.B).reshape(self.B, 1, 1, 1) if i < self.C else \
                info.mean_rgb[i-self.C].repeat(self.B).reshape(self.B, 1, 1, 1)
            std_i = info.std_in[i].repeat(self.B).reshape(self.B, 1, 1, 1) if i < self.C else \
                info.std_rgb[i-self.C].repeat(self.B).reshape(self.B, 1, 1, 1)
            cwl_i = info.cwl_in[i] if i < self.C else info.cwl_rgb[i-self.C]
            bw_i = info.bw_in[i] if i < self.C else info.bw_rgb[i-self.C]

            slice_x = slice_x.unsqueeze(1)
            slice_x = self.sub_mean(slice_x, mean_i, std_i)
            if i >= self.C:
                hr_rgb[:, i-self.C, :, :] = self.sub_mean(
                    hr_rgb[:, i-self.C, :, :].unsqueeze(1), mean_i, std_i).squeeze(1)

            slice_x = self.SFENet1(slice_x)
            slice_x = self.head(slice_x)

            weight1, bias1 = self.wlwp1(cwl_i, bw_i)
            slice_x = F.conv2d(slice_x, weight1, bias1, stride=1, padding=0)
            head_feat.append(slice_x)
        # ----------------------------------- Concate ----------------------------------- #
        f1 = torch.mean(torch.cat([feat.unsqueeze(0)
                                   for feat in head_feat], dim=0), dim=0)
        x = f1

        x = self.SFENet2(x)
        # ----------------------------------- Backbone ----------------------------------- #
        if self.args.body_blocks > 0:
            RDBs_out_body = []
            for i in range(self.args.body_blocks):
                x = self.RDBs_body[i](x)
                RDBs_out_body.append(x)

            x = self.GFF_body(torch.cat(RDBs_out_body, 1))
            x += f1
        # -------------------------------- Tail(Upsample) -------------------------------- #
        # Original parallel split-and-cat method
        # hsi_out = self.parallel_tail(x, hr_rgb, info.cwl_out, info.bw_out, info.mean_out, info.std_out)
        hsi_out = self.parallel_tail(x, hr_rgb, info.cwl_in, info.bw_in, info.mean_in, info.std_in)

        return hsi_out

    def parallel_tail(self, x, hr_rgb, cwl, bw, mean, std):
        out_images = []
        x = self.upsample(x)
        x = torch.cat([x, hr_rgb], 1)

        for i in range(len(cwl)):
            x_tail = x
            weight2, bias2 = self.wlwp2(cwl[i], bw[i])
            x_tail = F.conv2d(x_tail, weight2, bias2, stride=1, padding=0)
            out = self.tail(x_tail)
            out = self.add_mean(out, mean[i].repeat(self.B).reshape(
                self.B, 1, 1, 1), std[i].repeat(self.B).reshape(self.B, 1, 1, 1))
            out_images.append(out)
        out_images = torch.cat(out_images, dim=1)
        return out_images
