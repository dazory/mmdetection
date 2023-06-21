_base_ = ['../faster_rcnn_r50_fpn_1x_voc0712.py']

num_views = 2

### Model ###
model = dict(type='NViewsFasterRCNN')

### Data ###
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomColor', num_views=num_views),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
                               'img2', 'dx2']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))