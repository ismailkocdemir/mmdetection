from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class HDRDataset(CocoDataset):
    '''
    CLASSES = ("pottedplant",
                "person",
                "car",
                "bottle",
                "diningtable",
                "chair")
    '''
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tv/monitor')
     
