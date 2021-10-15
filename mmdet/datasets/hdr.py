from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class HDRDataset(CocoDataset):
    '''
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tv/monitor')
    '''
    CLASSES = (
             "pottedplant","person", "car", "bottle",
             "diningtable", "chair", "boat",
             "motorbike", "sofa", "tv/monitor",
             "aeroplane",  "dog",  "bicycle",
             "bus", "bird",   "horse",
             "train", "sheep", "cat", "cow"
            )