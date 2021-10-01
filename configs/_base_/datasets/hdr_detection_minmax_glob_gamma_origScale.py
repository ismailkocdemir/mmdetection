# dataset settings
dataset_type = 'HDRDataset'
data_root = '/truba/home/ikocdemir/data/HDR4RTT/0_RESIZED/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
min_max_config = dict(
    min_val=[-326.18848, -20.073975, -62.653442],
    max_val=[64033.875, 64785.125, 65504.0],
    gamma=True,
    rescale=65535.0
)
train_pipeline = [
    dict(type='LoadImageFromFile', hdr=True, min_max_norm=True, **min_max_config),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='RandomCrop', crop_size=[0.5, 0.5], crop_type='relative_range'),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(1024, 576)], keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', hdr=True, min_max_norm=True, **min_max_config),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 576),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
img_norm_gray = dict(
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
HDR_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, hdr=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 576),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_gray),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/instances_train2020_reduced.json',
            img_prefix=data_root + 'images/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instances_test2020_reduced.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instances_test2020_reduced.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    reference_HDR=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instances_test2020_reduced.json',
        img_prefix=data_root + 'images/',
        pipeline=HDR_pipeline)
    )
evaluation = dict(interval=1, metric='bbox')
