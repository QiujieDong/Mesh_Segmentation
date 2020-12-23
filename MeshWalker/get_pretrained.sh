#!/usr/bin/env bash

DATADIR='pretrained' #location where data gets downloaded to

echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/coseg_chairs-2.tar.gz
tar -xzvf coseg_chairs-2.tar.gz && rm coseg_chairs-2.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/coseg_vases.tar.gz
tar -xzvf coseg_vases.tar.gz && rm coseg_vases.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/coseg_aliens.tar.gz
tar -xzvf coseg_aliens.tar.gz && rm coseg_aliens.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/cubes.tar.gz
tar -xzvf cubes.tar.gz && rm cubes.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/human_seg.tar.gz
tar -xzvf human_seg.tar.gz && rm human_seg.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/shrec11_10-10_A.tar.gz
tar -xzvf shrec11_10-10_A.tar.gz && rm shrec11_10-10_A.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/shrec11_16-04_A.tar.gz
tar -xzvf shrec11_16-04_A.tar.gz && rm shrec11_16-04_A.tar.gz

wget https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/runs_pretrained/modelnet.zip
unzip modelnet.zip && rm modelnet.zip

