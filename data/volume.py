# -*- coding: utf-8 -*-
from torch.utils import data
from collections import OrderedDict
import os

from process.utils import parse_image_name


class Patient(data.Dataset):
    def __init__(self, data_file, select_attr):
        self.data_file = data_file
        self.select_attr = select_attr
        self.patients = OrderedDict()
        self.keys = []
        self.init()

    def init(self):
        info = dict()
        lines = [line.rstrip() for line in open(self.data_file, 'r')]

        for i, line in enumerate(lines):
            image, label = line.split()
            image_name = os.path.split(image)[1]
            mod, pid, _, _ = parse_image_name(image_name)
            if mod not in self.select_attr:
                continue
            if pid in self.patients.keys():
                self.patients[pid].append((image, label))
            else:
                self.patients[pid] = [(image, label)]
                self.keys.append(pid)
                if mod in info.keys():
                    info[mod] += 1
                else:
                    info[mod] = 1
        print('Load {} patient volumes!'.format(len(self.keys)))
        print(info)

    def __getitem__(self, idx):
        return self.patients[self.keys[idx]]

    def __len__(self):
        return len(self.keys)


if __name__ == '__main__':
    import SimpleITK as sitk
    import numpy as np

    pred = np.zeros((25, 25), dtype=np.int8)
    pred[:10, :10] = 1
    gt = np.zeros((25, 25), dtype=np.int8)
    gt[:12, :12] = 1

    dice_computer = sitk.LabelOverlapMeasuresImageFilter()
    dice_computer.Execute(gt, pred)
    print(dice_computer.GetDiceCoefficient())

