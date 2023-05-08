_base_ = [
    '/ws/external/configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py'
]

num_views = 2
use_clean = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
default_pipeline = [
    dict(type='PostNormalize', **img_norm_cfg, from_255=True, to_1=False),
    dict(type='PostPad', size_divisor=32)]
aug_pipeline = []

model = dict(
    type='CustomFasterRCNN',
    num_views=num_views,
    use_clean=use_clean,
    pipelines=[  # NOTE: must have same length as num_views
        default_pipeline,
        aug_pipeline + default_pipeline,
    ],
    hook_name_layer=dict(),
    additional_loss=dict(
        type='MultiViewLoss',
        criterion=dict(type='JSDLoss', target='before_branch', reduction='batchmean', name='loss_additional')),
)

''' Dataset '''
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline))
)

lr_config = dict(step=[1])  # [1] yields higher performance than [0]
runner = dict(type='EpochBasedRunner', max_epochs=2)