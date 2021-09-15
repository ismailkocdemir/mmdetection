_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/hdr_detection_absolute.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=20))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config =  dict(grad_clip=None) # dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6,7])
runner = dict(
    type='EpochBasedRunner', max_epochs=9)
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
