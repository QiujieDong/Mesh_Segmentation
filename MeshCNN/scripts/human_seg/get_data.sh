#!/usr/bin/env bash

DATADIR='/data/MeshCNN/' #location where data gets downloaded to /将数据放到SSD下

# get data
echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
wget https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz
tar -xzvf human_seg.tar.gz && rm human_seg.tar.gz