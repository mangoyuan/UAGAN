"""
Split the brain images into train-test.
Each patients contains only one modality.
"""
import os
import random
import argparse
import numpy as np
from utils import check_dir, parse_image_name
from PIL import Image


def split(brain_dir, save_dir, train_rate=6, n_split=3, exclude_threshold=400, n_max=None):
    patient_ids = np.array([f for f in os.listdir(brain_dir)])
    modalities = [m for m in os.listdir(os.path.join(brain_dir, patient_ids[0]))
                  if m != 'Label' and m != 'T1']

    perm = np.random.permutation(len(patient_ids))
    patient_ids = patient_ids[perm]
    id2mod = {p: modalities[i % len(modalities)] for i, p in enumerate(patient_ids)}

    train_r, test_r = train_rate, 10 - train_rate

    if not n_max:
        n_train = int(train_r / 10 * len(patient_ids))
        n_test = len(patient_ids) - n_train
    else:
        n_train = int(train_r / 10 * n_max)
        n_test = n_max - n_train

    splits = []
    i = 0
    for ns in range(n_split):
        if not n_max:
            splits.append(dict(test=patient_ids[i: i + n_test],
                               train=np.concatenate([patient_ids[:i], patient_ids[i + n_test:]])))
        else:
            splits.append(dict(test=patient_ids[i: i + n_test],
                               train=patient_ids[i + n_test:i + n_max]))
        i += n_test
        if i + n_test > len(patient_ids):
            perm = np.random.permutation(len(patient_ids))
            patient_ids = patient_ids[perm]
            i = 0

    check_dir(save_dir)
    for i in range(n_split):
        print(len(splits[i]['train']), len(splits[i]['test']))
        print(list(sorted(splits[i]['test'])))

        cls_rate = [0] * 4
        info = {x: {y: 0 for y in modalities} for x in ('train', 'test')}

        train_f = os.path.join(save_dir, '{}-train.txt'.format(i))
        test_f = os.path.join(save_dir, '{}-test.txt'.format(i))
        train_f = open(train_f, 'w')
        test_f = open(test_f, 'w')

        writer = dict(train=train_f, test=test_f)
        lines = dict(train=[[], []], test=[])
        for phase in ('train', 'test'):
            for pid in sorted(splits[i][phase]):
                save_mod = id2mod[pid]
                target_dir = os.path.join(brain_dir, pid, save_mod)
                for f in sorted(os.listdir(target_dir)):
                    m, pid, index, pn = parse_image_name(f)

                    fpath = os.path.join(target_dir, f)
                    lpath = os.path.join(brain_dir, pid, 'Label')
                    lpath = os.path.join(lpath, '{}_{}.png'.format(pid, index))

                    info[phase][m] += 1
                    lbl = np.array(Image.open(lpath))
                    image = np.array(Image.open(fpath))
                    if np.sum(image != 0) < exclude_threshold:
                        continue

                    cmap = [0, 80, 160, 240]
                    for c, cm in enumerate(cmap):
                        t = np.sum(lbl == cm)
                        cls_rate[c] += t

                    line = '{} {}\n'.format(fpath, lpath)
                    if phase == 'train':
                        lines[phase][int(pn)].append(line)
                    else:
                        lines[phase].append(line)

        for phase in ('train', 'test'):
            if phase == 'train':
                tar_list = lines['train'][0] + lines['train'][1]
                tar_list = list(sorted(tar_list))
            else:
                tar_list = lines['test']
            for line in tar_list:
                writer[phase].write(line)

        cls_rate = [cr / sum(cls_rate) for cr in cls_rate]
        print(info)
        # print(cls_rate)
        # break


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--brain_dir', type=str)
    parse.add_argument('--save_dir', type=str, default='.')
    parse.add_argument('--train_rate', type=int, default=5)
    parse.add_argument('--n_split', type=int, default=3)
    parse.add_argument('--n_max', type=int, default=100, help='Maximum number of dataset.')
    parse.add_argument('--seed', type=int, default=1234)
    parse.add_argument('--exclude_nolabel', type=int, default=1600,
                       help='Exclude images when #label smaller than threshold.')
    opt = parse.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)

    split(opt.brain_dir, opt.save_dir, train_rate=opt.train_rate, n_split=opt.n_split,
          exclude_threshold=opt.exclude_nolabel, n_max=opt.n_max)
    print('Done!')
