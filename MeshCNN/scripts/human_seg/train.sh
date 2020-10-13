#!/usr/bin/env bash

## run the training
"""
dataroot：数据集所在的路径
name:实验的名称，后续实验结果存储需要此名称，默认debug
arch:网络的选择,默认mconvnet
dataset_mode：数据集类型，默认为classification
ncf：
"""
python train.py \
--dataroot /data/MeshCNN/human_seg \
--name human_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 12 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \