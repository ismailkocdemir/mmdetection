#!/bin/bash

#python ../dataset_converters/cityscapes_extended.py /home/ihakki/h3dr/data/cityscapes --img-dir leftImg16bit -o annotations_50 -r 0.5
#python ../dataset_converters/cityscapes_extended.py /home/ihakki/h3dr/data/cityscapes --img-dir leftImg16bit -o annotations_60 -r 0.6
#python ../dataset_converters/cityscapes_extended.py /home/ihakki/h3dr/data/cityscapes --img-dir leftImg16bit -o annotations_70 -r 0.7
#python ../dataset_converters/cityscapes_extended.py /home/ihakki/h3dr/data/cityscapes --img-dir leftImg16bit -o annotations_80 -r 0.8
#python ../dataset_converters/cityscapes_extended.py /home/ihakki/h3dr/data/cityscapes --img-dir leftImg16bit -o annotations_90 -r 0.9

python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImgGamma/train
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImg16bitGamma/train/
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImgFattal/train
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImgDurand/train
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImgMantiuk/train
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImg8bit/train
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImgReinhardLocal/train
python -u bbox_analysis_Cityscapes.py /home/ihakki/h3dr/data/cityscapes/leftImgReinhardGlobal/train

