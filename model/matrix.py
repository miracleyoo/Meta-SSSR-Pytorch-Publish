import math
import torch
import torch.nn as nn


def matrix_2d_decode(matrix, inH, inW, scale, device, add_scale=True):
    matrix = matrix[0]
    scale_int = int(math.ceil(scale))
    h_offset = matrix[0][:inH*scale_int]
    w_offset = matrix[1][:inW*scale_int]

    # [outH, outW]: Every Row is the same
    h_offset_matrix = torch.cat(
        [h_offset.unsqueeze(-1)] * (inW*scale_int), 1)
    # [outH, outW]: Every Column is the same
    w_offset_matrix = torch.cat(
        [w_offset.unsqueeze(0)] * (inH*scale_int), 0)
    ref_matrix = torch.cat(
        [h_offset_matrix.unsqueeze(0), w_offset_matrix.unsqueeze(0)], 0)

    if add_scale:
        ref_matrix = torch.cat(
            [ref_matrix, torch.ones(1, (inH*scale_int), (inW*scale_int)).to(device)/scale])
    return ref_matrix.unsqueeze(0)


def repeat_x(x, scale):
    scale_int = math.ceil(scale)
    B, C, H, W = x.size()
    x = x.view(B, C, H, 1, W, 1)

    # [B, C, H, r, W, 1] -> [B, C, H, r, W, r] -> [B, r, r, C, H, W]
    x = torch.cat([x]*scale_int, 3)
    x = torch.cat([x]*scale_int, 5).permute(0, 3, 5, 1, 2, 4)

    # [B*r*r, C, H, W]
    return x.contiguous().view(-1, C, H, W)


def repeat_weight(weight, scale, inw, inh):
    k = int(math.sqrt(weight.size(0)))
    outw = inw * scale
    outh = inh * scale
    weight = weight.view(k, k, -1)
    scale_w = (outw+k-1) // k
    scale_h = (outh + k - 1) // k
    weight = torch.cat([weight] * scale_h, 0)
    weight = torch.cat([weight] * scale_w, 1)
    weight = weight[0:outh, 0:outw, :]
    return weight


def meta_upsample_1d(x, local_weight, scale, out_channel=1, verbose=False):
    """ Meta Upsample Module from the Meta-SR paper.
    "r" in this function's comment all means |r|, or ceil(r), which is an integer.

    Args:
        x: Input Tensor.
        local_weight: Local Weight.
        scale: The upscaling scale.
        out_channel: Output channel number.
        verbose: print debug info or not.
    """
    if verbose:
        print("scale:", scale, "\nlocal_weight shape 0:", local_weight.shape)

    # [N*r*r,inC,inH,inW]
    up_x = repeat_x(x, scale)

    # [N*r*r, C*k*k, inH*inW]
    cols = nn.functional.unfold(up_x, 3, padding=1)
    if verbose:
        print("Col Shape 0:", cols.shape)
    scale_int = math.ceil(scale)

    # [inH*r, inW*r, C*k*k]
    local_weight = repeat_weight(
        local_weight, scale_int, x.size(2), x.size(3))
    if verbose:
        print("local_weight shape 1:", local_weight.shape)

    # [N, r*r, C*k*k, inH*inW] -> [N, r*r, inH*inW, 1, C*k*k]
    cols = cols.view(cols.size(0)//(scale_int**2), scale_int**2,
                     cols.size(1), cols.size(2), 1).permute(0, 1, 3, 4, 2)
    if verbose:
        print("Col Shape 1:", cols.shape)

    # [r, r, inH, inW, C*k*k, outC]
    local_weight = local_weight.view(x.size(2), scale_int, x.size(
        3), scale_int, -1, out_channel).permute(1, 3, 0, 2, 4, 5).contiguous()
    if verbose:
        print("local_weight shape 2:", local_weight.shape)

    # [r*r, inH*inW, C*k*k, outC]
    local_weight = local_weight.view(
        scale_int**2, x.size(2)*x.size(3), -1, out_channel)
    if verbose:
        print("local_weight shape 3:", local_weight.shape)

    if verbose:
        print("Cols shape:", cols.shape,
              "\nLocal_weight shape:", local_weight.shape)

    # [N, r*r, inH*inW, 1, C*k*k].*[r*r, inH*inW, C*k*k, outC]
    # -> [N, r*r, inH*inW, 1, outC] -> [N, r*r, outC, inH*inW, 1]
    out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
    # [N, r*r, outC, inH*inW, 1] -> [N, r, r, outC, inH, inW] -> [N, outC, inH, r, inW, r]
    out = out.view(x.size(0), scale_int, scale_int, out_channel,
                   x.size(2), x.size(3)).permute(0, 3, 4, 1, 5, 2)
    # [N, outC, inH, r, inW, r] -> [N, outC, r*inH, r*inW]
    out = out.contiguous().view(x.size(0), out_channel,
                                scale_int*x.size(2), scale_int*x.size(3))
    return out


def meta_upsample_2d(x, local_weight, scale, out_channel=1, verbose=False):
    """ Meta Upsample Module from the Meta-SR paper.
    "r" in this function's comment all means |r|, or ceil(r), which is an integer.

    Args:
        x: Input Tensor. Shape: [N,inC,inH,inW]
        local_weight: Local Weight. Shape: [C*C'*k*k, H*|r|, W*|r|]
        scale: The upscaling scale.
        out_channel: Output channel number.
        verbose: print debug info or not.
    """
    N, inC, H, W = x.shape
    scale_int = math.ceil(scale)

    if verbose:
        print("scale:", scale, "\nlocal_weight shape 0:", local_weight.shape)

    # [N*r*r,inC,inH,inW]
    up_x = repeat_x(x, scale)

    # [N*r*r, C*k*k, inH*inW]
    cols = nn.functional.unfold(up_x, 3, padding=1)
    if verbose:
        print("Col Shape 0:", cols.shape)

    # [N, r*r, C*k*k, inH*inW] -> [N, r*r, inH*inW, 1, C*k*k]
    cols = cols.view(cols.size(0)//(scale_int**2), scale_int**2,
                     cols.size(1), cols.size(2), 1).permute(0, 1, 3, 4, 2)
    if verbose:
        print("Col Shape 1:", cols.shape)

    # [C*C'*k*k, H*r, W*r] -> (C', C*k*k, H, r, W, r) -> (r*r, HW, C*k*k, C')
    local_weight = local_weight.view(
        out_channel, -1, H, scale_int, W, scale_int).permute(3, 5, 2, 4, 1, 0)
    local_weight = local_weight.contiguous().view(
        scale_int**2, H*W, -1, out_channel)

    if verbose:
        print("local_weight shape 2:", local_weight.shape)

    if verbose:
        print("Cols shape:", cols.shape,
              "\nLocal_weight shape:", local_weight.shape)

    # [N, r*r, inH*inW, 1, C*k*k].*[r*r, inH*inW, C*k*k, outC]
    # -> [N, r*r, inH*inW, 1, outC] -> [N, r*r, outC, inH*inW, 1]
    out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
    # [N, r*r, outC, inH*inW, 1] -> [N, r, r, outC, inH, inW] -> [N, outC, inH, r, inW, r]
    out = out.view(x.size(0), scale_int, scale_int, out_channel,
                   x.size(2), x.size(3)).permute(0, 3, 4, 1, 5, 2)
    # [N, outC, inH, r, inW, r] -> [N, outC, r*inH, r*inW]
    out = out.contiguous().view(x.size(0), out_channel,
                                scale_int*x.size(2), scale_int*x.size(3))
    return out


def custom_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """ Customized Convolution using matrix operation
    """

    B, inC, in_h, in_w = x.shape
    outC, inC1, kh, kw = weight.shape
    assert inC == inC1
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if isinstance(stride, int):
        stride = (stride, stride)

    unfold = torch.nn.Unfold(kernel_size=(
        kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(x)
    # print("Shape of inp_unf:", inp_unf.transpose(1, 2).shape,
    #       "\nShape of weight:", weight.view(weight.size(0), -1).t().shape)

    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(
            weight.view(weight.size(0), -1).t()).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(
            weight.view(weight.size(0), -1).t()) + bias).transpose(1, 2)

    out_h = int((in_h + 2*padding[0] - dilation[0]
                 * (kh - 1) - 1)/stride[0] + 1)
    out_w = int((in_w + 2*padding[1] - dilation[1]
                 * (kw - 1) - 1)/stride[1] + 1)

    out = out_unf.contiguous().view(B, outC, out_h, out_w)
    return out


def center_wavelength_mapping(cwl, in_tensor, rgb_range, combine_type='add'):
    """ Map the center wavelength to a number in [0,1]
    """
    cwl = max(min(cwl - 200, 1000), 0)/1000
    map_tensor = cwl * torch.ones_like(in_tensor)
    if combine_type == 'add':
        return in_tensor + map_tensor
    elif combine_type == 'cat':
        return torch.cat((in_tensor, map_tensor), dim=1)
    else:
        raise KeyError('combine_type can only be chosen between add and cat!')
