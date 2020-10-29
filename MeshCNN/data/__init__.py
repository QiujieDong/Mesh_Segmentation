import torch.utils.data
from data.base_dataset import collate_fn


def CreateDataset(opt):
    """loads dataset class"""  # 选择导入数据的类别

    if opt.dataset_mode == 'segmentation':
        from data.segmentation_data import SegmentationData
        dataset = SegmentationData(opt)
    elif opt.dataset_mode == 'classification':
        from data.classification_data import ClassificationData
        dataset = ClassificationData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,  # 如果是True,那么每一次迭代都要进行数据重组
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)  # 将小batch中的tensor数据合并成一个list,当是map-dataset时使用

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data  # yield类似return
