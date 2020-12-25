#!/usr/bin/env bash

## run the training
python train.py \
--dataroot /data/COSEG/coseg_vases \
--name coseg_vases \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--lr 0.001 \
--batch_size 12 \
--num_aug 20 \
--slide_verts 0.2 \
--gpu_ids 3\


#
# python train.py --dataroot datasets/coseg_vases --name coseg_vases --arch meshunet --dataset_mode
segmentation --ncf 32 64 128 256 --ninput_edges 1500 --pool_res 1050 600 300 --resblocks 3 --lr 0.001 --batch_size 12 --num_aug 20