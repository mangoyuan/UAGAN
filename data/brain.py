# -*- coding: utf-8 -*-
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random

from utils import util, tsfm
from process.utils import parse_image_name


class Brain(data.Dataset):
    def __init__(self, data_file, selected_attrs, inputs_transform=None,
                 labels_transform=None, join_transform=None):
        self.selected_attrs = selected_attrs
        self.c_dim = len(self.selected_attrs)
        self.inputs_transform = inputs_transform
        self.labels_transform = labels_transform
        self.join_transform = join_transform
        self.data_file = data_file
        self.attr2idx = {}
        self.idx2attr = {}
        self.dataset = []
        self.init()

    def init(self):
        for i, attr_name in enumerate(self.selected_attrs):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = [line.rstrip() for line in open(self.data_file, 'r')]
        n = 0
        for i, line in enumerate(lines):
            image, label = line.split()
            image_name = os.path.split(image)[1]
            attr_name, pid, index, _ = parse_image_name(image_name)
            if attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                vec = [0] * self.c_dim
                vec[idx] = 1
                self.dataset.append([image, label, vec, idx, image_name])
                n += 1

        print('[*] Load {}, which contains {} images and labels, {}'.format(self.data_file,
                                                                            len(self.dataset), self.attr2idx))

    def __getitem__(self, idx):
        image, label, vec, idx, name = self.dataset[idx]
        images, labels = Image.open(image), Image.open(label)

        if self.join_transform:
            images, labels = self.join_transform(images, labels)
        if self.inputs_transform:
            images = self.inputs_transform(images)
        if self.labels_transform:
            labels = self.labels_transform(labels)
        return images, labels, torch.Tensor(vec), idx, name

    def __len__(self):
        return len(self.dataset)


def get_loaders(data_files, selected_attrs, batch_size=16, num_workers=1, image_size=224):
    train_join_tsfm = tsfm.Compose([
        tsfm.RandomHorizontalFlip(0.5),
        tsfm.RandomVerticalFlip(0.5),
        tsfm.Rotate(20),
        tsfm.Zoom(0.9, 1.1, image_size),
        tsfm.RandomCrop(image_size)
    ])

    input_tsfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])
    label_tsfm = T.Compose([
        tsfm.ToLongTensor()
    ])

    datasets = dict(train=Brain(data_files['train'], selected_attrs, inputs_transform=input_tsfm,
                                labels_transform=label_tsfm, join_transform=train_join_tsfm),
                    test=Brain(data_files['test'], selected_attrs, inputs_transform=input_tsfm,
                              labels_transform=label_tsfm, join_transform=None)
                    )
    loaders = {x: data.DataLoader(dataset=datasets[x], batch_size=batch_size, shuffle=(x == 'train'),
                                  num_workers=num_workers)
               for x in ('train', 'test')}
    return loaders

