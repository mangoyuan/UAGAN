import os
import numpy as np


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def pad_zero(s, length=3):
    s = str(s)
    assert len(s) < length + 1
    if len(s) < length:
        s = '0' * (length - len(s)) + s
    return s


def zscore(x):
    x = (x - x.mean()) / x.std()
    return x


def min_max(x):
    mi, ma = x.min(), x.max()
    x = (x - mi) / (ma - mi)
    return x


def percentile(x, prct):
    low, high = np.percentile(x, prct[0]), np.percentile(x, prct[1])
    x[x < low] = low
    x[x > high] = high
    return x


def parse_image_name(name):
    name = name.split('.')[0]
    mod, pid, index, pn = name.split('_')
    return mod, pid, index, pn


def center_crop(img, size):
    h, w = img.shape
    x, y = (h - size) // 2, (w - size) // 2
    img_ = img[x: x+size, y: y+size]
    return img_
