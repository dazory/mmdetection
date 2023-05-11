_base_ = [
    '../faster_rcnn_r50_fpn_1x_cityscapes.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='DeepAugment', model_name='EDSR', save_mode=True), ## -> image
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DeepAugment', model_name='CAE', save_mode=True, use_flip=False),  ## -> image
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data_root = '/ws/data/cityscapes/'
data = dict(
    workers_per_gpu=0,
    train=dict(
        dataset=dict(pipeline=train_pipeline)),
    val=dict(
        ann_file=data_root + 'annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix=data_root + 'leftImg8bit/train/',
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file=data_root + 'annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix=data_root + 'leftImg8bit/train/',
        pipeline=test_pipeline
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = None
