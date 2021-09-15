_base_ = [
    '../faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/hdr_detection_absolute.py',
]
'''
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])
runner = dict(
    type='EpochBasedRunner', max_epochs=12)
'''