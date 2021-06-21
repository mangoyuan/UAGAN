"""
Saving Training Data as Following Type.
./BrainTumour
    /1
        /FLAIR
        /T1
        /T1c
        /T2
        /Label
    /2
"""
import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from skimage import transform

from utils import check_dir, pad_zero, zscore, percentile, center_crop, min_max


def to_png(data_dir,
           save_dir,
           modality_order=('FLAIR', 'T1', 'T1c', 'T2'),
           cls_map=(0, 80, 160, 240),
           percent=None,
           z_score=True,
           crop=200,
           resize=128):
    check_dir(save_dir)
    image_dir = os.path.join(data_dir, 'imagesTr')

    for nii in os.listdir(image_dir):  # (H, W, D, M)
        if nii[0] == '.':
            continue
        patient_id = nii.split('.')[0].split('_')[1]
        fpath = os.path.join(image_dir, nii)
        volumes = nib.load(fpath).get_fdata()
        labels = nib.load(fpath.replace('imagesTr', 'labelsTr')).get_fdata()

        save_dir_pid = os.path.join(save_dir, patient_id)
        check_dir(save_dir_pid)
        save_dir_pid_label = os.path.join(save_dir_pid, 'Label')
        check_dir(save_dir_pid_label)
        for i, mo in enumerate(modality_order):  # (H, W, D)
            save_dir_pid_mo = os.path.join(save_dir_pid, mo)
            check_dir(save_dir_pid_mo)

            volume = volumes[..., i]
            if percent:
                volume = percentile(volume, percent)

            if z_score:
                volume = zscore(volume)
            volume = min_max(volume)

            for index in range(volume.shape[-1]):  # (H, W)
                slic = volume[..., index]
                slic = slic * 255
                label = labels[..., index]

                l_before = np.sum(label != 0)
                if crop:
                    slic = center_crop(slic, crop)
                    label = center_crop(label, crop)
                l_after = np.sum(label != 0)
                assert l_before - l_after == 0, l_before - l_after

                if resize:
                    slic = transform.resize(slic, (resize, resize), order=3, mode='constant')
                    label = transform.resize(label, (resize, resize), order=0, mode='constant')
                    slic[slic > 255] = 255
                    slic[slic < 0] = 0

                positive = 0 if np.sum(label != 0) == 0 else 1
                for cls, cm in enumerate(cls_map):
                    label[label == cls] = cls_map[cls]

                slic_name = '{}_{}_{}_{}.png'.format(mo, pad_zero(patient_id), pad_zero(index), positive)
                label_name = '{}_{}.png'.format(pad_zero(patient_id), pad_zero(index))
                slic = Image.fromarray(slic.astype('uint8'))
                label = Image.fromarray(label.astype('uint8'))

                slic.save(os.path.join(save_dir_pid_mo, slic_name))
                save_l_name = os.path.join(save_dir_pid_label, label_name)
                if not os.path.exists(save_l_name):
                    label.save(os.path.join(save_dir_pid_label, label_name))
                print('Saving slice in to {}'.format(slic_name))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--brain_dir', type=str, default='/mnt/sda1/medical/Task01_BrainTumour')
    parse.add_argument('--save_dir', type=str, default='/mnt/sda1/theoyuan/BrainTumour128')
    parse.add_argument('--modality_order', nargs='+', default=['FLAIR', 'T1', 'T1c', 'T2'])
    parse.add_argument('--cls_map', type=int, nargs='+', default=['0', '80', '160', '240'])
    parse.add_argument('--percent', type=float, nargs='+', default=None)
    parse.add_argument('--crop', type=int, default=200)
    parse.add_argument('--resize', type=int, default=128)
    parse.add_argument('--zscore', type=int, default=1)

    opt = parse.parse_args()
    cls_map = [int(cm) for cm in opt.cls_map]
    if opt.percent:
        percent = [float(p) for p in opt.percent]
    else:
        percent = opt.percent
    to_png(opt.brain_dir, opt.save_dir, modality_order=opt.modality_order, cls_map=cls_map,
           percent=percent, z_score=opt.zscore, crop=opt.crop, resize=opt.resize)
    print('Done!')

