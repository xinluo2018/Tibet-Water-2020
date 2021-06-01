'''simple pre-processing'''
import torch
import numpy as np
import random
from skimage.transform import downscale_local_mean


class toTensor:
    """Convert ndarrays to torch.Tensors (channel, height, width)."""
    def __call__(self, image, truth):
        image = image.astype(np.float32).transpose((2, 0, 1)) #（channels, height, width）
        image = torch.from_numpy(image).float()
        truth = torch.from_numpy(truth).long()
        return image, truth

class normalize:
    '''normalization with the given per-band max and min values'''
    def __init__(self, max, min):
        '''max, min: list, values corresponding to each band'''
        self.max, self.min = max, min
    def __call__(self, image, truth):
        for band in range(image.shape[0]):
            image[band, :,:] = (image[band,:,:]-self.min[band])/(self.max[band]-self.min[band]+0.0001)
        return image, truth

class crop:
    '''randomly crop corresponding to specific size'''
    def __init__(self, size=(256,256)):
        self.size = size
    def __call__(self, image, truth):
        '''size: (height, width)'''
        start_h = random.randint(0, truth.shape[0]-self.size[0])
        start_w = random.randint(0, truth.shape[1]-self.size[1])
        image = image[:,start_h:start_h+self.size[0], start_w:start_w+self.size[1]]
        truth = truth[start_h:start_h+self.size[0], start_w:start_w+self.size[1]]
        return image, truth

class crop_scales:
    '''randomly crop corresponding to specific size'''
    def __init__(self, scales=(2048, 512, 256)):
        self.scales = scales
    def __call__(self, image, truth):
        '''scales: tuple (high -> low)'''
        patches = []
        scales_ratio = [scale//self.scales[-1] for scale in self.scales]
        start_high_row = random.randint(0, truth.shape[0]-self.scales[0])
        start_high_col = random.randint(0, truth.shape[1]-self.scales[0])
        patch_high = image[:,start_high_row:start_high_row+self.scales[0], \
                            start_high_col:start_high_col+self.scales[0]]
        patches.append(patch_high)
        for scale in self.scales[1:]:
            start_offset = (self.scales[0]-scale)//2
            patch_low = patch_high[:, start_offset:start_offset+scale, \
                                            start_offset:start_offset+scale]
            patches.append(patch_low)
        start_offset_lowest = (self.scales[0]-self.scales[-1])//2
        truth = truth[start_high_row+start_offset_lowest:start_high_row+start_offset_lowest+self.scales[-1], \
                    start_high_col+start_offset_lowest:start_high_col+start_offset_lowest+self.scales[-1]]
        patches = [downscale_local_mean(patch, (1, ratio, ratio)) for patch, ratio in zip(patches, scales_ratio)]
        patches = [torch.from_numpy(patch).float() for patch in patches ]   
        return patches, truth

