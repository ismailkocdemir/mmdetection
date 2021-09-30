# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/home/ihakki/h3dr/data/cityscapes/'
img_norm_cfg = dict(
    mean = [64.29153202, 75.54099511, 65.06069988],
    std =  [46.66449826, 50.68927987, 45.54992053],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.5),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations_16bit/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImgFattal/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations_16bit/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImgFattal/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations_16bit/instancesonly_filtered_gtFine_test.json',
        img_prefix=data_root + 'leftImgFattal/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
