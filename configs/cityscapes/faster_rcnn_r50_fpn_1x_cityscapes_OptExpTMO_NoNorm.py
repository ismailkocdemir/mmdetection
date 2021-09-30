__base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_scratch.py',
    '../_base_/datasets/cityscapes_detection_OptExp.py',
    '../_base_/default_runtime.py'
]


model = dict(
    type='FasterRCNN',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        zero_init_residual=False,
        no_norm=True)
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None 
workflow = [('train', 1)]

evaluation = dict(interval=1, classwise=True)
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='SGD',
    lr=0.04,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[11])
runner = dict(
    type='EpochBasedRunner', max_epochs=13)
log_config = dict(
            interval=100, 
            hooks=[
                    dict(type='TensorboardLoggerHook'),
                    dict(type='TextLoggerHook'),
                ]
)
work_dir = "/home/ihakki/h3dr/experiments/faster_rcnn_optexp/run_scratch_8"
gpu_ids = range(0, 1)
