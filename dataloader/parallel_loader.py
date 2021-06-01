import sys
sys.path.append( "/home/yons/Desktop/developer-luo/SWatNet")
import torch
import threading as td
from queue import Queue
from utils.transforms import crop_scales
from utils.img_aug import rotate, flip, noise, missing
import gc

## define multi-thread job
def job(q, scene_list, truth_list):    
    '''q is Queue'''
    patch_list, ptruth_list = [],[]
    transforms = [rotate(p=1), flip(p=0.5), noise(p=0.5, \
            std_min=0.001, std_max=0.1), missing(p=0.5, ratio_max = 0.25)]
    '''convert image to patches group'''
    zip_data = list(zip(scene_list, truth_list))
    for scene, truth in zip_data:
        patches_group, truth=crop_scales(scales=[2048, 512, 256])(scene, truth)
        for transform in transforms:
            patches_group, truth = transform(patches_group, truth)
        truth = torch.unsqueeze(truth,0)
        patch_list.append(patches_group), ptruth_list.append(truth)
    q.put((patch_list, ptruth_list))


def parallel_read(scene_list, truth_list, num_thread=20):
    '''multi-thread reading training data
        cooperated with the job function
    '''
    patch_lists, ptruth_lists = [], []
    q = Queue()
    threads = [td.Thread(target=job, args=(q, scene_list, \
                        truth_list)) for i in range(num_thread)]
    start = [t.start() for t in threads]
    join = [t.join() for t in threads]
    for i in range(num_thread):
        patch_list, ptruth_list = q.get()
        patch_lists += patch_list
        ptruth_lists += ptruth_list
    return patch_lists, ptruth_lists


class parallel_load_dset(torch.utils.data.Dataset):
    '''patch and the truth parallel reading from RAM memory
        patch_list, truth_list are lists (consist of torch.tensor).
        time record(num_work=0): data (750 patches) read->11 s, model train -> 3 s
    '''
    def __init__(self, scene_list, truth_list, num_thread):

        self.scene_list = scene_list
        self.truth_list = truth_list 
        self.num_thread = num_thread
        self.patches_list, self.ptruth_list = parallel_read(\
                                        scene_list, truth_list, num_thread)
    def __getitem__(self, index): 
        '''load patches and truths'''
        patch = self.patches_list[index]
        truth = self.ptruth_list[index]
        if index == len(self.patches_list)-1:        ## update the dataset
            del self.patches_list, self.ptruth_list
            gc.collect()
            self.patches_list, self.ptruth_list = parallel_read(\
                        self.scene_list, self.truth_list, self.num_thread)
        return patch, truth

    def __len__(self):
        return len(self.patches_list)

