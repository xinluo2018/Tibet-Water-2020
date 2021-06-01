from utils.transforms import toTensor, normalize
from utils.tiff_io import readTiff
import numpy as np

s1_min = [-57.78, -70.37, -58.98, -68.47]
s1_max = [25.98, 10.23, 29.28, 17.60]

def read_preprocess(paths_as, paths_des, paths_truth):
    scene_list, truth_list = [],[]
    for i in range(len(paths_as)):
        ascend_src, ascend = readTiff(paths_as[i])
        descend_src, descend = readTiff(paths_des[i])
        truth_src,truth = readTiff(paths_truth[i])
        scene = np.concatenate((ascend, descend), axis=-1)
        scene, truth = toTensor()(scene, truth)
        scene, truth = normalize(max=s1_max, min=s1_min)(scene, truth)
        scene_list.append(scene), truth_list.append(truth)
    return scene_list, truth_list

