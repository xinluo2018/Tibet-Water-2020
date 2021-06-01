'''!!! note: all the input patch are multi-scale patches'''
import torch
import random

class rotate:
    '''randomly rotation with given probability'''
    def __init__(self, p=0.5):
        self.p = p 
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() < self.p: 
            degree = random.randint(1,3)
            patches = [torch.rot90(patch, degree, [1, 2]) for patch in patches]
            truth = torch.rot90(truth, degree, [0, 1])
        return patches, truth

class flip:
    '''randomly flip with given probability '''
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() < self.p:
            if random.random() < 0.5:
                for i in range(len(patches)): # up <-> down
                    patches[i] = torch.from_numpy(patches[i].numpy()[:,::-1,:].copy()) 
                truth = torch.from_numpy(truth.numpy()[::-1,:].copy())
            elif random.random() < 0.5:       # left <-> right
                for i in range(len(patches)):
                    patches[i] = torch.from_numpy(patches[i].numpy()[:,:,::-1].copy()) 
                truth = torch.from_numpy(truth.numpy()[:,::-1].copy())  
        return patches, truth

class noise:
    '''add randomly noise with given randomly standard deviation'''
    def __init__(self, p=0.5, std_min=0.001, std_max =0.1):
        self.p = p
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() < self.p:
            for i in range(len(patches)):
                std = random.uniform(self.std_min, self.std_max)
                noise = torch.normal(mean=0, std=std, size=patches[i].size())
                patches[i] = patches[i].add(noise)
        return patches, truth

class missing:
    '''randomly stripe missing'''
    def __init__(self, p=0.5, ratio_max = 0.25):
        self.p = p
        self.ratio_max = ratio_max
    def __call__(self, patches, truth):
        if random.random() < self.p:
            h_img = truth.shape[0]
            miss_max = h_img*self.ratio_max
            for i in range(len(patches)):
                h_miss = random.randint(0, miss_max)
                w_miss = random.randint(0, miss_max)
                start_h = random.randint(0, h_img-h_miss)
                start_w = random.randint(0, h_img-h_miss)
                patches[i][:,start_h:start_h+h_miss, start_w:start_w+w_miss] = 0
        return patches, truth


