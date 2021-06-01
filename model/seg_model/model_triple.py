import sys
import os 
import torch.nn as nn
sys.path.append(os.path.dirname(__file__))



def convert_g_l(img_global, scale_factor):
    '''global_size should be divisible by local_size.
        input img_global is a 4-d (n,c,h,w) tensor.
    '''
    size_g = img_global.shape[2]
    size_l = size_g//scale_factor
    local_row_start = (size_g - size_l)//2
    img_local = img_global[:,:,local_row_start:local_row_start\
                            +size_l, local_row_start:local_row_start+size_l]
    img_local = nn.functional.interpolate(img_local, \
                            scale_factor=scale_factor, mode='nearest')
    return img_local

