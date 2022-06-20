from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        dataset_class = _DATASETS[data['dataset_class']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if dataset_class == VOCDataset:
            args['keep_difficult'] = not is_train
        elif dataset_class == COCODataset:
            args['remove_empty'] = is_train
        dataset = dataset_class(**args) # args : {'data_dir': 'datasets/VOC2007', 'split': 'trainval', 'transform': <ssd.data.transforms.transforms.Compose object at 0x7fa701c80eb0>, 'target_transform': <ssd.data.transforms.target_transform.SSDTargetTransform object at 0x7fa701c80e80>, 'keep_difficult': False}
        datasets.append(dataset) # VOC : 2007 & VOC 2012
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0] # check 
    if len(datasets) > 1: # CONCAT VOC 2007 & 2012 DATASETS
        dataset = ConcatDataset(datasets)

    return [dataset]
