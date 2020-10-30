import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
import numpy as np
from models.layers.mesh import Mesh


class SegmentationData(BaseDataset):  # 创建seg的dataloader

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)  # the mesh list of train
        self.seg_paths = self.get_seg_files(
            self.paths, os.path.join(self.root, 'seg'), seg_ext='.eseg')  # 返回一个以seg_ext结尾的新的list，这个list是对应的label文件
        self.sseg_paths = self.get_seg_files(
            self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
        self.classes, self.offset = self.get_n_segs(
            os.path.join(self.root, 'classes.txt'), self.seg_paths)  # 返回从0开始的类别列表
        self.nclasses = len(self.classes)
        self.size = len(self.paths)  # Tne len of mesh for training
        self.get_mean_std()  # return the mean, std and ninput_channels
        # # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True,  # TODO:read the class of Mesh
                    export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        label = read_seg(self.seg_paths[index]) - self.offset  # label的设定
        # 若label与ninput_edges的数目不匹配，则用-1进行padding
        label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
        meta['label'] = label
        soft_label = read_sseg(self.sseg_paths[index])  # 返回一个包含1的array
        meta['soft_label'] = pad(
            soft_label, self.opt.ninput_edges, val=-1, dim=0)
        # get edge features
        edge_features = mesh.extract_features()  # 返回feature
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(
                os.path.basename(path))[0] + seg_ext)
            assert(os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    # 人为的将seg划分为几个class，比如对human_seg数据分为八个class,分别为[头，上身，大腿, 小腿，脚，上臂，小臂，手]
    @staticmethod
    def get_n_segs(classes_file, seg_files):  # 让类别从0开始
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                # np.concatenate()同时多个数组的拼接，
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)  # 去除重复的信息
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path):  # 输出train中mesh文件的所有路径
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        # os.walk()输出目录中的文件名；sorted()在list上的排序，默认升序。
        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes


def read_seg(seg):  # 数据读取
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)  # 若labels>0，则为1
    return sseg_labels
