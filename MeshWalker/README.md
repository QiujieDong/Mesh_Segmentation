# MeshWalker: Deep Mesh Understanding by Random Walks
<img src='/doc/images/teaser_fig.png'>

## SIGGRAPH ASIA 2020 [[Paper]](https://arxiv.org/abs/2006.05353)
Created by [Alon Lahav](mailto:alon.lahav2@gmail.com).

This repository contains the implementation of [MeshWalker](https://arxiv.org/abs/2006.05353).

## Installation
A step-by-step installation guide for Ubuntu is provided in [INSTALL.md](./INSTALL.md).

## Data
Note for this README: each time `<dataset>` is mentioned, 
it should be replaced by one of the following:

```
1. modelnet40
2. engraved_cubes
3. shrec11
4. coseg
5. human_seg
```

<img src='/doc/images/segmentaion_edges_human_body.gif'>

### Raw datasets
To get the raw datasets go to the relevant website, 
and put it under `MeshWalker/datasets_raw/<dataset>`. 
- [ModelNet](https://modelnet.cs.princeton.edu/)
  (Right click on `ModelNet40.zip`, to download the dataset. 
- [Engraved Cubes](https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz) (from [MeshCNN](https://ranahanocka.github.io/MeshCNN/) website).
- [Human-seg17](https://www.dropbox.com/sh/cnyccu3vtuhq1ii/AADgGIN6rKbvWzv0Sh-Kr417a?dl=0) (from [Toric Covers](https://github.com/Haggaim/ToricCNN) website).
- [COSEG](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm).
- Shrec11 - to be added later.

You can also download it from our [raw_datasets](https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_raw/) folder.


### Processed
To prepare the data, run `python dataset_prepare.py <dataset>`

Or download the data after processing from 
[datasets_processed](https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshWalker/mesh_walker_data/datasets_processed/)
to `MeshWalker/datasets_processed/<dataset>`. 
Processing will rearrange dataset in `npz` files, labels included, vertex niebours added.

Use the following to download all:
```
bash ./get_datasets.sh
```
 
## Training
```
python train_val.py <job> <part>
```
While `<job>` can be one of the following: 
`shrec11` / `coseg` / `human_seg` / `cubes` / `modelnet40`.
`<job>` can also be `all` to run all of the above.

`<part>` should be used in case of `shrec11` or `coseg` datasets.
For `shrec11` it should be one of the follows: 
`10-10_A` / `10-10_B` / `10-10_C` / `16-04_A` / `16-04_B` / `16-04_C`.

For `coseg` it should be one of the follows: `aliens` / `vases` / `chairs`.

You will find the results at: `MeshWalker\runs\???`

Use tensorboard to show training results: `tensorboard <trained-model-folder>`

Note that "accuracy" tab is a fast accuracy calculated while training, 
it is not the final accuracy we get using averaging.
To get the final accuracy results, please refer to the "full_accuracy" tab at tensorboard, 
or run evaluation scripts.

<img src='/doc/images/2nd_fig.png'>

## Evaluating
After training is finished (or pretrained is downloaded), 
to evaluate **segmentation** model run: 
```
python evaluate_segmentation.py <job> <part> <trained model directory>
```
For example:
```
python evaluate_segmentation.py coseg chairs pretrained/coseg_chairs/
```
Or:
```
python evaluate_segmentation.py human_seg --- pretrained/0010-15.11.2020..05.25__human_seg/
```

To evaluate **classification** model run: 
```
python evaluate_segmentation.py <job> <part> <trained model directory>
```

`<job>` and `<part>` are define the same as in `train_val.py`. 

## Pretrained   
You can use some pretrained models from [our pretrained folder](https://technionmail-my.sharepoint.com/personal/alon_lahav_campus_technion_ac_il/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Falon%5Flahav%5Fcampus%5Ftechnion%5Fac%5Fil%2FDocuments%2Fmesh%5Fwalker%2Fpretrained)  
to run evaluation only.

Or download them all using
```
bash ./get_pretrained.sh
``` 

## Reference
If you find our code or paper useful, please consider citing:
```
@article{lahav2020meshwalker,
  title={MeshWalker: Deep Mesh Understanding by Random Walks},
  author={Lahav, Alon and Tal, Ayellet},
  journal={arXiv preprint arXiv:2006.05353},
  year={2020}
}
```

## Questions / Issues
If you have questions or issues running this code, please open an issue.