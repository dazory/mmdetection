_base_ = ['/ws/external/configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py']

## Model ###
# learning policy
lr_config = dict(step=[1]) # [1] yields higher performance than [0]
runner = dict(type='EpochBasedRunner', max_epochs=2)  # actual epoch = 2 * 8 = 16
