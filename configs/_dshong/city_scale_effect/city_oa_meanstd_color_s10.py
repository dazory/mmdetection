_base_ = ['../faster_rcnn_r50_fpn_1x_cityscapes__e2.py']

num_views = 2
use_oa=True
oa_version='meanstd'
use_mix=True
mixture_coeff=-1 # oa_dist mode
severity=10
name = f"city-oa_{oa_version}_color_s{severity}-faster_rcnn_fpn"

WANDB_ENTITY = "kaist-url-ai28"
WANDB_PROJECT_NAME = "mmdetection_oa"


### Model ###
model = dict(
    type='NViewsFasterRCNN',
    layer_name_type_list=[('roi_head.bbox_roi_extractor', 'input'),
                          ('roi_head.bbox_roi_extractor', 'module'),
                          ('roi_head.bbox_roi_extractor', 'output')],
    do_something=dict(type='ScaleEffect', )
)

### Data ###
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Color', num_views=num_views, severity=severity,
         use_oa=use_oa, oa_version=oa_version,
         use_mix=use_mix, mixture_coeff=mixture_coeff),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'img2', 'dx2']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))

### Logger ###
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='CustomMMDetWandbHook',
             init_kwargs={
                 "entity": WANDB_ENTITY, "project": WANDB_PROJECT_NAME,
                 "name": name,
             },
             interval=50,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100,
             bbox_score_thr=0.3)
    ])