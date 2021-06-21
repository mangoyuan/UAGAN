# Unified Attentional Generative Adversarial Network for Brain Tumor Segmentation From Multimodal Unpaired Images

## Installation
- Run on python3.6, Pytorch0.4 and CUDA 8.0.
- Install [tensorboardX](https://github.com/lanpa/tensorboardX).
- Clone this repo.

## Data Preparation
1. Download the `Task01_BrainTumour` dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
2. Pre-process, save as `png` files and split train-test list.
```bash
cd process

python to_png.py --brain_dir /path/to/Task01_BrainTumour \
--save_dir /path/to/png_dataset \
--crop 200 --resize 128

python split.py --brain_dir /path/to/png_dataset \
--save_dir .
```

## Train
All model will stop at `max_epoch` and make predictions at the last epoch.
```bash
cd ..
./uagan.sh
```

## Acknowledgement
Part of the code is revised from
- [yunjey/stargan](https://github.com/yunjey/stargan)
- [JorisRoels/domain-adaptive-segmentation](https://github.com/JorisRoels/domain-adaptive-segmentation)
