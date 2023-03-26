## author: xin luo
## creat: 2021.9.8; modify: 2023.2.4
## des: implement surface water mapping by using trained GMNet.
## how to use:
##  functional api:
##    wat_pred = gmnet_infer(rsimg, path_model_w, orbit)
##  command line:
##    python gmnet_infer.py -m path_model_weights -img path/of/data/*.tif -orbit as -o directory/of/output -s 1
##  !!note: 1. the .tif format image should be the original value, or scale * original value
##          2. if the input s1_img orbit is as_des, the ascending and descending image should layer stacked firstly. 

import os
import sys
sys.path.append("/home/xin/Developer-luo/Tibet-Water-2020")   ## change to your project path
import numpy as np
import argparse
import torch
from utils.imgPatch import imgPatch
import scipy.ndimage
import gc

from utils.geotif_io import readTiff, writeTiff
from model.seg_model.unet_scales_gate import unet_scales_gate

# ### ----------- default path of the pretrained watnet model ----------- 
path_gmnet_w = ['model/trained_model/scales/unet_scales_gate/traset/as_des/train_0_weights.pth']
path_gmnet_w_as = ['model/trained_model/scales/unet_scales_gate/traset/as/train_0_weights.pth']
path_gmnet_w_des = ['model/trained_model/scales/unet_scales_gate/traset/des/train_0_weights.pth']


## ------------- pre-defined functions -------------
### get arguments
def get_args():

    description = 'surface water mapping by using pretrained gmnet'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-m', '--model', type=str, nargs='+', 
        default=path_gmnet_w, 
        help=('pretrained gmnet model (based on both ascending and descending images) weight(torch, .pth)'))
    parser.add_argument(
        '-img', '--ifile_img', type=str, nargs='+', 
        default=[None],
        help=('path of s1 image file(s) to process (.tiff)'))
    parser.add_argument(
        '-orbit', '--orbit', type=str, nargs='+', 
        default=['as'], 
        help=('orbit of the s1 image: including as, des, as_des'))
    parser.add_argument(
        '-o', '--odir', type=str, nargs='+', 
        default=[None], 
        help=('directory to write'))
    parser.add_argument(
        '-s', '--scale_DN', type=int, nargs='+', 
        default=[None], 
        help=('DN scale of the sentinel-1 image, if None, the scale can be regard as 1'))

    return parser.parse_args()

def normalize(s1_img, s1_min, s1_max, orbit):
    '''normalization'''
    s1_img_nor = s1_img.copy()    
    for band in range(s1_img.shape[-1]):
        s1_img_nor[:,:,band] = (s1_img_nor[:,:,band] - s1_min[band])/(s1_max[band]-s1_min[band]+0.0001)
    s1_img_nor = np.clip(s1_img_nor, 0., 1.) 
    s1_img_nor[np.isnan(s1_img_nor)]=0         # remove nan value
    return s1_img_nor

def img2patchin(img, scales = [256, 512, 2048], overlay=60):
    ratio_mid, ratio_high = scales[1]//scales[0], scales[2]//scales[0],
    imgPat_ins = imgPatch(img=img, patch_size=scales[0], edge_overlay = overlay)
    patch_low_list = imgPat_ins.toPatch()
    patch_mid_list = imgPat_ins.higher_patch_crop(higher_patch_size=scales[1])
    patch_high_list = imgPat_ins.higher_patch_crop(higher_patch_size=scales[2])
    '''---- Resize multi-scale patches to the same size'''
    patch_mid2low_list = [scipy.ndimage.zoom(input=patch, zoom=(1/ratio_mid, \
                                            1/ratio_mid, 1), order=0) for patch in patch_mid_list]
    patch_high2low_list = [scipy.ndimage.zoom(input=patch, zoom=(1/ratio_high, \
                                            1/ratio_high, 1), order=0) for patch in patch_high_list]
    return patch_low_list, patch_mid2low_list, patch_high2low_list, imgPat_ins

def gmnet_infer(rsimg, path_model_w, orbit):
    '''
    params
        rsimg: np.array()
        path_model_w: the pretrained gmnet weights, (format .pth)
        orbit: as/des/as_des.
    return:
        the prediction water map
    '''
    ### ------------- super parameters ------------- 
    s1_min = [-57.78, -70.37, -58.98, -68.47]   # as-vv, as-vh, des-vv, des-vh
    s1_max = [25.98, 10.23, 29.28, 17.60]       # as-vv, as-vh, des-vv, des-v

    # ------------- Device --------------- #
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    '''Model loading'''
    if orbit == 'as_des':
        model = unet_scales_gate(num_bands=4, num_classes=2,scale_high=2048, scale_mid=512,scale_low=256)
    elif orbit == 'as':
        model = unet_scales_gate(num_bands=2, num_classes=2,scale_high=2048, scale_mid=512,scale_low=256)
        s1_min=s1_min[0:2]; s1_max=s1_max[0:2]
    elif orbit == 'des':
        model = unet_scales_gate(num_bands=2, num_classes=2,scale_high=2048, scale_mid=512,scale_low=256)
        s1_min=s1_min[2:4]; s1_max=s1_max[2:4]

    model.load_state_dict(torch.load(path_model_w))
    if device:
        model.to(device); model = model.eval()
    print('...pretrained models have been loaded')

    ### ---- data normalization ---
    rsimg_nor = normalize(rsimg, s1_min, s1_max, orbit)
    del rsimg; gc.collect()
    print('...Input image have been normalized')

    ### ---- image to patch ----
    print('--- Convert input image to patch list...')
    patch_low_list, patch_mid_list, patch_high_list, imgPat_ins = \
                        img2patchin(rsimg_nor, scales = [256, 512, 2048], overlay=60)
    del rsimg_nor; gc.collect()
    ## formating data from 3d to 4d torch.tensor
    patch_high_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).\
                                                    float() for patch in patch_high_list]
    patch_mid_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).\
                                                    float() for patch in patch_mid_list]
    patch_low_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).\
                                                    float() for patch in patch_low_list]
    inputs = tuple(zip(patch_high_list_, patch_mid_list_, patch_low_list_))
    del patch_high_list, patch_mid_list, patch_low_list
    del patch_high_list_, patch_mid_list_, patch_low_list_; gc.collect()
    num_inputs = len(inputs)
    print('Number of multi-scale patches:', num_inputs)
    print('...Input image have been converted to patches list')

    ### ---- surface water mapping ----
    print('--- surface water mapping using the pretrained gmnet model...')
    pred_patch_list = []
    with torch.no_grad():
        for idx in range(num_inputs):
            if device:
                in_data = [inp.to(device) for inp in inputs[idx]]
            pred_patch = model(in_data)                 
            pred_patch = pred_patch.cpu()
            pred_patch_list.append(pred_patch)     # convert from gpu to cpu
        
    pred_patch_list = [np.squeeze(patch, axis = 0).permute(1, 2, 0) for patch in pred_patch_list]
    pro_map = imgPat_ins.toImage(pred_patch_list)
    wat_map = np.where(pro_map>0.5, 1, 0)
    return wat_map

if __name__ == '__main__':

    args = get_args()
    path_gmnet_w = args.model[0]
    ifiles = args.ifile_img   # list
    orbit = args.orbit[0]
    odir = args.odir[0]
    scale_DN = args.scale_DN[0]

    ''' Obtain input s1 image file.'''
    io_files = []
    if odir: dir_wat = odir
    else: dir_wat = '/'.join(ifiles[0].split('/')[:-2]) + '/s1-water'
    if not os.path.exists(dir_wat):
        os.makedirs(dir_wat)

    ### pair-wise input image and output result.
    for id, i_file in enumerate(ifiles):   
        name_wat = i_file.split('/')[-1].split('.')[0] + '_water.tif'
        out_wat_path = dir_wat + '/' + name_wat
        io_files.append((i_file, out_wat_path))   


    ''' Loop for each file '''
    for i in range(len(io_files)):

        ### ---- 1. preprocessing
        ### ---- 1.1 data reading
        print('--- data reading...')
        s1_img, s1_info = readTiff(path_in = io_files[i][0])                
        if scale_DN: s1_img = np.float32(s1_img)/scale_DN
        s1_img[s1_img==0]=['nan']

        print('...Water map have been produced')
        wat_map = gmnet_infer(s1_img, path_gmnet_w, orbit)
        ### ---- 4. write out the water map -----
        print('--- write out the result image...')
        writeTiff(im_data = wat_map.astype(np.uint8), 
                    im_geotrans = s1_info['geotrans'], 
                    im_geosrs = s1_info['geosrs'], 
                    path_out = io_files[i][-1])
        print('--- write out -->')
        print(io_files[i][-1])
        del wat_map; gc.collect()


