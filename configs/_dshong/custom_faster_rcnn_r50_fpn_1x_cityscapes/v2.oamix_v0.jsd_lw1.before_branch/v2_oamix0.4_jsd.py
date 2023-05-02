_base_ = [
    '../custom_faster_rcnn_r50_fpn_1x_cityscapes.py',
]

num_views = 2
use_clean = True

''' Post-transforms '''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
default_pipeline = [
    dict(type='PostNormalize', **img_norm_cfg, from_255=True, to_1=True),  # NOTE: `to_rgb` is not supported yet
    dict(type='PostPad', size_divisor=32)
]
aug_pipeline = [
    dict(type='OAMix', version='0.4', mixture_depth=1),
    dict(type='PostNormalize', **img_norm_cfg, from_255=False, to_1=True),  # NOTE: `to_rgb` is not supported yet
    dict(type='PostPad', size_divisor=32)
]

hook_name = 'before_branch'
hook_name_layer = {hook_name: 'roi_head.bbox_head.shared_fcs.1'}

model = dict(
    num_views=num_views,
    use_clean=use_clean,
    pipelines=[  # NOTE: must have same length as num_views
        default_pipeline,
        aug_pipeline,
    ],
    hook_name_layer=hook_name_layer,
    additional_loss=dict(type='MultiViewLoss',
                         criterion=dict(type='JSDLoss', target=hook_name, reduction='batchmean', name='loss_additional')),
)
