_base_ = [
    '../faster_rcnn_r50_fpn_1x_coco.py',
    '../../_base_/datasets/hdr_detection.py',
]
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10])
runner = dict(
    type='EpochBasedRunner', max_epochs=20)
