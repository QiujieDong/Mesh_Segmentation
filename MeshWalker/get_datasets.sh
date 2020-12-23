#!/usr/bin/env bash

DATADIR='datasets_processed' #location where data gets downloaded to

echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_processed/coseg_from_meshcnn.tar.gz
tar -xzvf coseg_from_meshcnn.tar.gz && rm coseg_from_meshcnn.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_processed/cubes.tar.gz
tar -xzvf cubes.tar.gz && rm cubes.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_processed/human_seg_from_meshcnn.tar.gz
tar -xzvf human_seg_from_meshcnn.tar.gz && rm human_seg_from_meshcnn.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_processed/modelnet40.tar.gz
tar -xzvf modelnet40.tar.gz && rm modelnet40.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_processed/shrec11.tar.gz
tar -xzvf shrec11.tar.gz && rm shrec11.tar.gz
