
# root directory
root = "/home/yons/Desktop/developer-luo/SWatNet"
import sys
sys.path.append(root)
from utils.transforms import crop_scales
from utils.img_aug import rotate, flip, noise, missing


##----------Training parameter------- ##
epoch = 100
lr = 0.001
batch_size = 16

##---------data pre-processing-------- ##
s1_min = [-57.78, -70.37, -58.98, -68.47]
s1_max = [25.98, 10.23, 29.28, 17.60]

transforms_tra = [
            crop_scales(scales=(2048, 512, 256)),
            rotate(p=1), 
            flip(p=0.5), 
            noise(p=0.5, std_min=0.001, std_max =0.1), 
            missing(p=0.5, ratio_max = 0.25)
            ]
transforms_test = [
            crop_scales(scales=(2048, 512, 256)),
            ]

