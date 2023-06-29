_base_ = ['/ws/external/configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py']

name = "city_faster-rcnn"

WANDB_ENTITY = 'kaist-url-ai28'
WANDB_PROJECT_NAME = 'mmdetection_oa'

## Model ###
# learning policy
lr_config = dict(step=[1]) # [1] yields higher performance than [0]
runner = dict(type='EpochBasedRunner', max_epochs=2)  # actual epoch = 2 * 8 = 16

### Logger ###
log_config = dict(
    dict(type='TextLoggerHook'),
    hooks=[
        dict(type='MMDetWandbHook',
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