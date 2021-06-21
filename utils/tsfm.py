# -*- coding: utf-8 -*-
import numpy as np
import torch
import PIL
from scipy import ndimage
from skimage.transform import resize
from skimage.util import random_noise
import random
import elasticdeform
import numbers
import torchvision.transforms.functional as F


# Transforms for PIL.Image
class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl


class Rotate:
    def __init__(self, degrees):
        self.degrees = (-degrees, degrees)

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, lbl):
        angle = self.get_params(self.degrees)
        img = F.rotate(img, angle, 3)
        lbl = F.rotate(lbl, angle, 0)
        return img, lbl


class Zoom:
    def __init__(self, low, high, min_size):
        self.scales = (low, high)
        self.min_size = min_size

    @staticmethod
    def get_params(scales):
        scale = random.uniform(scales[0], scales[1])
        return scale

    def __call__(self, img, lbl):
        scale = self.get_params(self.scales)
        size = int(scale * img.size[0])
        if size < self.min_size:
            size = self.min_size
        img = F.resize(img, size, 3)
        lbl = F.resize(lbl, size, 0)
        return img, lbl


# Others
class ToLongTensor:
    def __call__(self, lbl):
        lbl = np.array(lbl, dtype='uint8')
        lbl[lbl != 0] = 1
        return torch.from_numpy(lbl).long()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

