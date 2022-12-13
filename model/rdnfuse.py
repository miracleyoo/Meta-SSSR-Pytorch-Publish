from model import common
from functools import partial

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

## ---------------------- RDB Modules ------------------------ ##


class RDB_Conv(nn.Module):
    """ Residual Dense Convolution.
    """

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    """ Residual Dense Block.
    """

    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale
        G0 = args.n_feats
        kSize = 3
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.mean_rgb = torch.FloatTensor(self.args.mean_rgb).to(self.device)
        self.std_rgb = torch.FloatTensor(self.args.std_rgb).to(self.device)

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        self.sub_mean = partial(common.mean_shift_2d, add=False)
        self.add_mean = partial(common.mean_shift_2d, add=True)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            args.n_colors+args.n_hc, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, int(G * r * r), kSize,
                          padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(int(r)),
                nn.Conv2d(G, G, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        m_tail = [
            nn.Conv2d(
                G+3, args.n_hc, kSize,
                padding=(kSize-1)//2
            )
        ]

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, lr_rgb, hr_rgb, info):
        B, C, *_ = x.shape
#         print(x.shape, lr_rgb.shape, mean_in.shape)

        x = self.sub_mean(x, info.mean_in, info.std_in)
        lr_rgb = self.sub_mean(lr_rgb, self.mean_rgb.repeat(
            B, 1).to(x), self.std_rgb.repeat(B, 1).to(x))
        hr_rgb = self.sub_mean(hr_rgb, self.mean_rgb.repeat(
            B, 1).to(x), self.std_rgb.repeat(B, 1).to(x))

        x = torch.cat((x, lr_rgb), axis=1)

        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        x = self.UPNet(x)
        x = self.tail(torch.cat([x, hr_rgb], 1))
        x = self.add_mean(x, info.mean_in, info.std_in)
        return x
