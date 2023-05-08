# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .custom_two_stage import CustomTwoStageDetector


@DETECTORS.register_module()
class CustomFasterRCNN(CustomTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 num_views=1,
                 use_clean=True,
                 pipelines=None,
                 **kwargs
                 ):
        super(CustomFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            num_views=num_views,
            use_clean=use_clean,
            pipelines=pipelines,
            **kwargs
        )
