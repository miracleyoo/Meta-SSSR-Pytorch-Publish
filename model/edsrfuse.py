from model import common
from functools import partial

import torch
import torch.nn as nn


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return EDSR(args, dilated.dilated_conv)
    else:
        return EDSR(args)


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.mean_rgb = torch.FloatTensor(self.args.mean_rgb).to(self.device)
        self.std_rgb = torch.FloatTensor(self.args.std_rgb).to(self.device)
        

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = partial(common.mean_shift_2d, add=False)
        self.add_mean = partial(common.mean_shift_2d, add=True)

        # define head module
        m_head = [conv(args.n_colors+args.n_hc, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        self.up = common.Upsampler(conv, scale, n_feats, act=False)
        m_tail = [
            nn.Conv2d(
                n_feats+3, args.n_hc, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, lr_rgb, hr_rgb, mean_in=None, std_in=None):
        B, C, *_ = x.shape

        x = self.sub_mean(x, mean_in, std_in)
        lr_rgb = self.sub_mean(lr_rgb, self.mean_rgb.repeat(
            B, 1).to(x), self.std_rgb.repeat(B, 1).to(x))
        hr_rgb = self.sub_mean(hr_rgb, self.mean_rgb.repeat(
            B, 1).to(x), self.std_rgb.repeat(B, 1).to(x))

        x = torch.cat((x, lr_rgb), axis=1)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.up(res)

        x = self.tail(torch.cat((x, hr_rgb), axis=1))
        x = self.add_mean(x, mean_in, std_in)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
