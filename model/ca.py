from torch import nn


class ECALayer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    """ Squeeze and Excite Layer

    Args:
        channel: Number of channels of the input feature map
        reduction: SE layer internel channel size reduction number.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SEResBlock(nn.Module):
    """ Squeeze and Excite Residual Block

    Args:
        conv: Convolutional Layer(block) used in this block.
        channel: Number of channels of the input feature map.
        kernel_size: Convolutional Layer kernel size.
        reduction: SE layer internel channel size reduction number.
        bias: Whether to use bias in conv.
        bn: Whether to use batch normalization or not.
        act: Activation Layer
    """

    def __init__(
            self, conv, channel, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True)):

        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(channel, channel, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(channel))
            if i == 0:
                modules_body.append(act)
        modules_body.append(SELayer(channel, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res
