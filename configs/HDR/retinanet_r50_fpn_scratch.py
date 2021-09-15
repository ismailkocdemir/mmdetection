_base_ = [
    '../_base_/models/retinanet_r50_fpn_scratch.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=20))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) #dict(grad_clip=None) # 
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 15])
runner = dict(
    type='EpochBasedRunner', max_epochs=18)
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
load_from = None #'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
