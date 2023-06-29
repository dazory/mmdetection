_base_ = ['/ws/external/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py']

name = "voc_faster-rcnn"

WANDB_ENTITY = "kaist-url-ai28"
WANDB_PROJECT_NAME = "mmdetection_oa"


# learning policy
lr_config = dict(policy='step', step=[3]) # actual epoch = 3 * 3 = 9
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

### Logger ###
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
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