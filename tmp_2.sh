#!/bin/bash

path_model_weights='model/trained_model/scales/unet_scales_gate/traset/as/train_0_weights.pth' 
rsimg='data/dset/s1_ascend_clean/scene01_s1as.tif'
dir_out='./'
python scripts/gmnet_infer.py -m $path_model_weights -img $rsimg -orbit as -o $dir_out -s 1

