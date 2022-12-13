import os
import glob

import json
import pickle
import imageio
import random
import numpy as np


def check_and_load(f, verbose=True):
    if verbose:
        print('Loading {}...'.format(f))
    with open(f, 'rb') as _f:
        ret = pickle.load(_f)
    return ret

def get_mean_std(data_dir, data_name):
    assert 'bin' in data_dir
    channels = 31
    channel_list = []
    for i in range(channels):
        channel_list.append([])
    for i in os.listdir(data_dir):
        if 'SRF' in i:
            continue
        # output = {"mean":{},"std":{}}
        if 'pt' in i:
            f = os.path.join(data_dir,i)
            f_image = check_and_load(f)
            image = f_image[0]['image']
            #print(image.shape)
            for ch in range(image.shape[2]):
                channel_list[ch].append(image[:,:,ch].flatten())
    output_mean = []
    output_std = []
    for ch in channel_list:
        ch_all = np.concatenate(ch)
        ch_mean = np.mean(ch_all)
        ch_std = np.std(ch_all)
        output_mean.append(ch_mean)
        output_std.append(ch_std)
    # output_mean = np.stack(output_mean,axis=2)
    # output_std = np.stack(output_std,axis=2)
    out = {'std':output_std,'mean':output_mean}
    with open(data_name+'.json', 'w') as _f:
        json.dump(out, _f)    
        #         cur_mean = np.mean(image[:,:,ch])
        #         cur_std = np.std(image[:,:,ch])
        #         output['mean'][ch+1] = cur_mean
        #         output['std'][ch+1] = cur_std
        # total_out[i] = output
    return out
    #with open(data_name+'.json','w') as fout:
        #json.dump(total_out,fout)
            #print(image.shape)
            #image_list.append(image)

data_dir = "/mnt/nfs/scratch1/zhongyangzha/Meta-FM-SR-Pytorch/datasets/NTIRE_VAL/bin/HR"
#"/mnt/nfs/scratch1/zhongyangzha/Meta-FM-SR-Pytorch/datasets/NTIRE/bin/HR"
#"/mnt/nfs/scratch1/zhongyangzha/Meta-FM-SR-Pytorch/datasets/DIV2K/bin/HR"
#"/mnt/nfs/scratch1/zhongyangzha/Meta-FM-SR-Pytorch/datasets/NTIRE_VAL/bin/HR"
data_name = "NTIRE_VAL"
#"DIV2K"
#"NTIRE"
stat = get_mean_std(data_dir, data_name)
print(stat)
