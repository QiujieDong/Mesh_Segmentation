from __future__ import print_function
import torch
import numpy as np
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


MESH_EXTENSIONS = [
    '.obj',
]  # 定义mesh文件的扩展名为obj文件


# endswith() 方法用于判断字符串是否以指定后缀结尾，any()全部为 False，则返回 False，如果有一个为 True，则返回 True。
def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)


def pad(input_arr, target_length, val=0, dim=1):  # padding操作。
    shp = input_arr.shape  # label的size
    npad = [(0, 0) for _ in range(len(shp))]  # 与label的size一致的（0，0）矩阵
    # 这里的padding只在行的后面进行padding，array的前面不进行padding
    npad[dim] = (0, target_length - shp[dim])
    # pad_width(1,2)表示在一维数组array前面填充1位，最后面填充2位
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


def seg_accuracy(predicted, ssegs, meshes):  # 计算分割精度
    correct = 0
    # squeeze()将输入张量中的一维张量去除，这里-1是表示若tensor最后一个维度是1，那么将其去掉。索引从0开始
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
        edge_areas = torch.from_numpy(mesh.get_edge_areas())
        correct += (correct_vec.float() * edge_areas).sum()
    return correct


def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy
