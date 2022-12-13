import math
import numpy as np
import torch

def input_matrix_wpn_2d(inH, inW, scale, add_scale=True):
    outH, outW = int(scale * inH), int(scale * inW)
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH*scale_int)
    w_offset = torch.ones(inW*scale_int)

    mask_h = torch.zeros(inH*scale_int)
    mask_w = torch.zeros(inW*scale_int)

    # projection  coordinate  and caculate the offset
    # [outH,]
    h_project_coord = torch.arange(0, outH, 1).mul(1.0 / scale)
    int_h_project_coord = torch.floor(h_project_coord)

    # [outH,]
    offset_h_coord = h_project_coord - int_h_project_coord  # v_h
    int_h_project_coord = int_h_project_coord.int()  # floor_h

    number = 0
    temp = -1
    for i in range(len(offset_h_coord)):
        if int_h_project_coord[i] != temp:
            number = int_h_project_coord[i]*scale_int
            temp = int_h_project_coord[i]
        else:
            number += 1
        mask_h[number] = 1
        h_offset[number] = offset_h_coord[i]

    # [outW,]
    w_project_coord = torch.arange(0, outW, 1).mul(1.0 / scale)
    int_w_project_coord = torch.floor(w_project_coord)

    # [outW,]
    offset_w_coord = w_project_coord - int_w_project_coord  # v_w
    int_w_project_coord = int_w_project_coord.int()  # floor_w

    number = 0
    temp = -1
    for i in range(len(offset_w_coord)):
        if int_w_project_coord[i] != temp:
            number = int_w_project_coord[i]*scale_int
            temp = int_w_project_coord[i]
        else:
            number += 1
        mask_w[number] = 1
        w_offset[number] = offset_w_coord[i]

    # [outH, outW]: Every Row is the same
    h_offset_matrix = torch.cat(
        [h_offset.unsqueeze(-1)] * (inW*scale_int), 1)
    # [outH, outW]: Every Column is the same
    w_offset_matrix = torch.cat(
        [w_offset.unsqueeze(0)] * (inH*scale_int), 0)

    mask_h = torch.cat([mask_h.unsqueeze(-1)] * (scale_int * inW),
                       1).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w.unsqueeze(0)] * (scale_int * inH),
                       0).view(-1, scale_int * inW, 1)
    mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(
        scale_int * inH, scale_int * inW)
    mask_mat = mask_mat.eq(2)

    ref_matrix = torch.cat(
        [h_offset_matrix.unsqueeze(0), w_offset_matrix.unsqueeze(0)], 0)

    if add_scale:
        ref_matrix = torch.cat(
            [ref_matrix, torch.ones(1, (inH*scale_int), (inW*scale_int))/scale])
    return ref_matrix.unsqueeze(0), mask_mat


def input_matrix_wpn_1d(inH, inW, scale, add_scale=True):
    '''
    By given the scale and the size of input image, we caculate the
    input matrix for the weight prediction network
    Args:
        inH, inW: the size of the feature maps
        scale: is the upsampling times
    '''
    outH, outW = int(scale * inH), int(scale * inW)
    # mask records which pixel is invalid, 1 valid or 0 invalid
    # h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    # print(f"inH:{inH}, outH:{outH}, scale_int:{scale_int}, ")
    # [inH, r, 1]
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH, scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)

    # projection  coordinate  and caculate the offset
    # [outH,]
    h_project_coord = torch.arange(0, outH, 1).mul(1.0 / scale)
    int_h_project_coord = torch.floor(h_project_coord)

    # [outH,]
    offset_h_coord = h_project_coord - int_h_project_coord  # v_h
    int_h_project_coord = int_h_project_coord.int()  # floor_h

    # [outW,]
    w_project_coord = torch.arange(0, outW, 1).mul(1.0 / scale)
    int_w_project_coord = torch.floor(w_project_coord)

    # [outW,]
    offset_w_coord = w_project_coord - int_w_project_coord  # v_w
    int_w_project_coord = int_w_project_coord.int()  # floor_w

    # flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag, 0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    # print(f"==> h offset shape:{h_offset.shape}")
    flag = 0
    number = 0
    """ Shape:[inW, |r|]
    [[1, 1, 1, 0]
        [1, 1, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 1, 1]
        [1, 1, 1, 0]...]
    """
    for i in range(outW):
        if int_w_project_coord[i] == number:
            # First line case and the [1:] case for the other lines
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            # The first element for every line except the first line
            # Second 0 in the next line is actually the flag=0 case
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    # [inH, |r|, |r|*inW] -> [|r|*inH, |r|*inW, 1]: Every Line is the same
    h_offset_coord = torch.cat(
        [h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)

    # print(f"h_offset_coord shape:{h_offset_coord.shape}")
    # [|r|* inH, inW, |r|] -> [|r|*inH, |r|*inW, 1]: Every Column is the same
    w_offset_coord = torch.cat(
        [w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW),
                       2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH),
                       0).view(-1, scale_int * inW, 1)

    # [|r|* inH, |r|*inW, 2]
    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    # print(f"pos_mat shape:{pos_mat.shape}")

    mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(
        scale_int * inH, scale_int * inW)
    mask_mat = mask_mat.eq(2)

    i = 1
    h, w, _ = pos_mat.size()
    while(pos_mat[i][0][0] >= 1e-6 and i < h):
        i = i+1

    j = 1
    # pdb.set_trace()
    h, w, _ = pos_mat.size()
    while(pos_mat[0][j][1] >= 1e-6 and j < w):
        j = j+1

    pos_mat_small = pos_mat[0:i, 0:j, :]
    # print(f"pos_mat_small shape: {pos_mat_small.shape}")

    pos_mat_small = pos_mat_small.contiguous().view(1, -1, 2)
    if add_scale:
        scale_mat = torch.zeros(1, 1)
        scale_mat[0, 0] = 1.0 / scale
        # (inH*inW*scale_int**2, 4)
        scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(1)), 0)
        pos_mat_small = torch.cat(
            (scale_mat.view(1, -1, 1), pos_mat_small), 2)

    # outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
    # print(f"pos_mat_small shape: {pos_mat_small.shape}")
    return pos_mat_small, mask_mat

    # speed up the model by removing the computation