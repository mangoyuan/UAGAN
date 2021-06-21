#!/usr/bin/env bash

TRAIN_L=process/0-train.txt
TEST_L=process/0-test.txt
CHECKPOINT=./checkpoint

# Train.
python train_uagan.py --train_list ${TRAIN_L} --phase train \
--test_list ${TEST_L} --beta1 0.\
--checkpoint_dir ${CHECKPOINT} \
--lambda_seg 100 --lambda_shape 100 \
--max_epoch 100 --decay_epoch 40 --shape_epoch 60 \
--selected_attr FLAIR T1c T2 | tee uagan.log

# Infer.
python train_uagan.py --test_list ${TEST_L} --phase test \
--train_list ${TRAIN_L} \
--checkpoint_dir ${CHECKPOINT} --batch_size 1 \
--test_epoch 100 --use_tensorboard 0 \
--selected_attr FLAIR T1c T2
