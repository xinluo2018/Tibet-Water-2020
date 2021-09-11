## author: xin luo, 
## created: 2021.7.8
## des: data augmentation before model training.

import torch
import random
import copy
import numpy as np
import torchvision.transforms as transforms

class numpy2tensor:
    '''des: np.array to torch.Tensor
    '''
    def __call__(self, patch_group, ptruth):
        if isinstance(patch_group,list):
            patch_group_tensor = [torch.from_numpy(patch.copy()).float() for patch in patch_group]
        else:
            patch_group_tensor = torch.from_numpy(patch_group.copy()).float()
        ptruth_tensor = torch.from_numpy(ptruth.copy())
        return patch_group_tensor, ptruth_tensor

#### ------------- numpy-based ---------------
#### -----------------------------------------

class rotate:
    '''numpy-based
       des: randomly rotation with given probability
       '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        # else:
        k = random.randint(1,3)
        if isinstance(patches,list):
            patches_rot = [np.rot90(patch, k, [1, 2]) for patch in patches]
        else:
            patches_rot = np.rot90(patches, k, [1, 2])
        truth_rot = np.rot90(truth, k, [0, 1])
        return patches_rot, truth_rot

class flip: 
    '''numpy-based
       des: randomly flip with given probability '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''input image, truth are np.array'''
        if random.random() > self.p:
            return patches, truth
        if random.random() < 0.5:            # up <-> down
            if isinstance(patches, list):
                patches_flip = [np.flip(patch, 1) for patch in patches] 
            else:
                patches_flip = np.flip(patches, 1)
            truth_flip = np.flip(truth, 0)
        else:                                # left <-> right
            if isinstance(patches, list):
                patches_flip = [np.flip(patch, 2) for patch in patches]
            else:
                patches_flip = np.flip(patches, 2)
            truth_flip = np.flip(truth, 1)
        return patches_flip, truth_flip

class missing:
    '''numpy-based
       des: randomly stripe missing'''
    def __init__(self, prob=0.5, ratio_max = 0.25):
        self.p = prob
        self.ratio_max = ratio_max
    def __call__(self, patches, truth):
        if random.random() > self.p:
            return patches, truth
        # else:
        patches_miss = copy.deepcopy(patches)
        h_img = truth.shape[0]
        miss_max = int(h_img*self.ratio_max)
        if isinstance(patches, list):
            for i in range(len(patches)):
                h_miss = random.randint(0, miss_max)
                w_miss = random.randint(0, miss_max)
                start_h = random.randint(0, h_img-h_miss)
                start_w = random.randint(0, h_img-h_miss)
                patches_miss[i][:,start_h:start_h+h_miss, start_w:start_w+w_miss] = 0
        else: 
            h_miss = random.randint(0, miss_max)
            w_miss = random.randint(0, miss_max)
            start_h = random.randint(0, h_img-h_miss)
            start_w = random.randint(0, h_img-h_miss)
            patches_miss[:,start_h:start_h+h_miss, start_w:start_w+w_miss] = 0

        return patches_miss, truth

class noise:
    '''numpy-based 
       !!! slower than torch-based 
       des: add randomly noise with given randomly standard deviation
    '''
    def __init__(self, prob=0.5, std_min=0.001, std_max=0.1):
        self.p = prob
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        else:
            if isinstance(patches, list):
                patches_noisy = []
                for i in range(len(patches)):
                    std = random.uniform(self.std_min, self.std_max)
                    noise = np.random.normal(loc=0, scale=std, size=patches[i].shape)
                    patches_noisy.append(patches[i]+noise)
            else:
                std = random.uniform(self.std_min, self.std_max)
                noise = np.random.normal(loc=0, scale=std, size=patches.shape)
                patches_noisy = patches + noise
            return patches_noisy, truth

class colorjitter:
    '''numpy-based
       des: randomly colorjitter with given probability, 
       color jitter contains bright adjust and contrast adjust.
       color jitter is performed for per band.
       '''
    def __init__(self, prob=0.5, alpha=0.1, beta=0.1, t=0.1):
        self.p = prob
        self.alpha = alpha
        self.beta = beta
        self.t = t
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth        
        if isinstance(patches, list):
            num_band = patches[0].shape[0]
            patches_cat = np.concatenate(patches, 0)
            patches_aug = []
            for i in range(patches_cat.shape[0]):
                alpha = random.uniform(1-self.alpha, 1+self.alpha)
                beta = random.uniform(-self.beta, self.beta)
                t = random.uniform(-self.t, self.t)
                band_aug = alpha*(patches_cat[i:i+1]-t)+beta
                band_aug = np.clip(band_aug, 0, 1)
                patches_aug.append(band_aug)
            patches_aug = np.concatenate(patches_aug, 0)
            patches_aug = [patches_aug[0:num_band],patches_aug[num_band:2*num_band],patches_aug[2*num_band:]]
        else: 
            patches_aug = []
            for i in range(patches.shape[0]):                
                alpha = random.uniform(1-self.alpha, 1+self.alpha)
                beta = random.uniform(-self.beta, self.beta)
                t = random.uniform(-self.t, self.t)
                band_aug = alpha*(patches[i:i+1]-t)+beta
                band_aug = np.clip(band_aug, 0, 1)
                patches_aug.append(band_aug)
            patches_aug = np.concatenate(patches_aug, 0)
        return patches_aug, truth

class bandjitter:
    '''numpy-based
       des: randomly bandjitter with given probability, 
       '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth  
        if isinstance(patches, list):
            num_band = patches[0].shape[0]
        else: 
            num_band = patches.shape[0]
        order = np.arange(num_band)
        random.shuffle(order)
        if isinstance(patches, list):
            patches_aug = [patch[order] for patch in patches]
        else: 
            patches_aug = patches[order]
        return patches_aug, truth



#### -------------- torch-based ----------------
#### -------------------------------------------
class torch_rotate:
    '''torch-based
       des: randomly rotation with given probability
       '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        k = random.randint(1,3)
        if isinstance(patches,list):
            patches_rot = [torch.rot90(patch, k, [1, 2]) for patch in patches]
        else: 
            patches_rot = torch.rot90(patches, k, [1, 2])
        truth_rot = torch.rot90(truth, k, [0, 1])
        return patches_rot, truth_rot

class torch_noise:
    '''torch-based: faster than numpy-based
       des: add randomly noise with given randomly standard deviation
    '''
    def __init__(self, prob=0.5, std_min=0.001, std_max =0.1):
        self.p = prob
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        if isinstance(patches,list):
            patches_noisy = []
            for i in range(len(patches)):
                std = random.uniform(self.std_min, self.std_max)
                noise = torch.normal(mean=0, std=std, size=patches[i].size())
                patches_noisy.append(patches[i].add(noise))
        else:
            std = random.uniform(self.std_min, self.std_max)
            noise = torch.normal(mean=0, std=std, size=patches.size())
            patches_noisy = patches.add(noise)
        return patches_noisy, truth


class torch_colorjitter:
    '''torch-based colorjitter, it is different from color_bias
       des: randomly colorjitter with given probability
       color jitter contains bright adjust, contrast adjust, saturation and hue adjust
       color jitter is performed for per band.
       '''
    def __init__(self, prob=0.5):
        self.p = prob
        self.toPIL = transforms.ToPILImage()
        self.colorjit = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.toTensor = transforms.ToTensor()
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        if isinstance(patches, list):
            num_band = patches[0].shape[0]
            patches_cat = torch.cat(patches, 0)
            patches_aug = []
            for i in range(patches_cat.shape[0]):
                patch_coljit = self.toPIL(patches_cat[i:i+1])
                patch_coljit = self.colorjit(patch_coljit)
                patch_coljit = self.toTensor(patch_coljit)
                patches_aug.append(patch_coljit)
            patches_aug = torch.cat(patches_aug, 0)
            patches_aug = [patches_aug[0:num_band],patches_aug[num_band:num_band+4],patches_aug[num_band+4:]]
        else: 
            patches_aug = []
            for i in range(patches.shape[0]):
                patch_coljit = self.toPIL(patches[i:i+1])
                patch_coljit = self.colorjit(patch_coljit)
                patch_coljit = self.toTensor(patch_coljit)
                patches_aug.append(patch_coljit)
            patches_aug = torch.cat(patches_aug, 0)
        return patches_aug, truth
