import torch.utils.data as data
import numpy as np
import pickle  # 实现python对象的持久化存储
import os


class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        super(BaseDataset, self).__init__()

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(
            self.root, 'mean_std_cache.p')  # 计算平均值与标准差并保存起来
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    # TODO:这里面用了太多的print,使用必要的print,其他的使用log
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']  # 只取features
                # 对每一个特征计算mean,然后相加，得到这个特征的整体mean。std也是一样。
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)  # 对mean取平均,也就是每一个特征的mean
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)  # 将python中的对象序列化成二进制对象
            print('saved: ', mean_std_cache)
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)  # 读取给定的二进制对象数据，并将其转换为 Python 对象；
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})
    return meta
