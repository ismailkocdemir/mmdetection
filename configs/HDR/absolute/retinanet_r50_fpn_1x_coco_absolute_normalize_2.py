_base_ = [
    '../retinanet_r50_fpn_1x_coco.py',
    '../../_base_/datasets/hdr_detection_absolute_normalize.py',
]


# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None) # dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12])
runner = dict(
    type='EpochBasedRunner', max_epochs=16)
