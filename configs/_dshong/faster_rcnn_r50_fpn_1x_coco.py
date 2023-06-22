_base_ = ['/ws/external/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

name = "coco_faster-rcnn"

WANDB_ENTITY = 'kaist-url-ai28'
WANDB_PROJECT_NAME = 'mmdetection_oa'

### Logger ###
log_config = dict(
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