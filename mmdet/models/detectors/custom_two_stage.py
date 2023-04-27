# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_loss
from .two_stage import TwoStageDetector
from ...datasets.pipelines import Compose


@DETECTORS.register_module()
class CustomTwoStageDetector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 num_views=1,
                 use_clean=True,
                 pipelines=None,
                 hook_name_layer=None,
                 additional_loss=None,
                 *args, **kwargs,
                 ):
        super(CustomTwoStageDetector, self).__init__(*args, **kwargs)
        self.num_views = num_views
        self.use_clean = use_clean
        assert pipelines is not None, 'pipelines must be specified'
        self.pipelines = [Compose(pipeline) for pipeline in pipelines]

        # hooks
        self.hook_data = dict()
        self.handles = []
        self.hook_name_layer = hook_name_layer if hook_name_layer is not None else dict()
        self._register_hook(self.hook_name_layer)

        # additional loss
        self.additional_loss = build_loss(additional_loss) if additional_loss is not None else None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ''' post-transforms'''
        imgs = []
        for i in range(self.num_views):
            imgs.append(self.pipelines[i](img))
        imgs = torch.cat(imgs, dim=0)


        ''' The first view is forwarded '''
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        ''' The rest views are forwarded '''
        x_rest = self.extract_feat(imgs[1:])

        # RPN forward and loss
        gt_bboxes_rest = gt_bboxes * (self.num_views - 1)
        gt_labels_rest = gt_labels * (self.num_views - 1)
        gt_bboxes_ignore_rest = sum(gt_bboxes_ignore, []) if gt_bboxes_ignore is not None else gt_bboxes_ignore
        gt_masks_rest = sum(gt_masks, []) if gt_masks is not None else gt_masks
        img_metas_rest = sum([img_metas] * (self.num_views - 1), [])
        _ = self.roi_head.forward_train(x_rest, img_metas_rest, proposal_list,
                                                 gt_bboxes_rest, gt_labels_rest,
                                                 gt_bboxes_ignore_rest, gt_masks_rest,
                                                 **kwargs)

        if self.additional_loss is not None:
            additional_loss = self.additional_loss(self.hook_data)
            losses.update(additional_loss)

        self._remove_hook_data()

        return losses

    def _find_layer(self, model, layer_name):
        '''
        Input:
                layer_name = (str)
        Example:
                layer_name = 'rpn.rpn_cls_layer'
                layer = find_layer(model, layer_name)
                print(layer)
        '''
        if isinstance(layer_name, str):
            layer_name = layer_name.split('.')

        if len(layer_name) != 0:
            for name, child in model.named_children():
                if name == layer_name[0]:
                    if len(layer_name) == 1:
                        return child
                    child = self._find_layer(child, layer_name[1:])
                    if child != None:
                        return child
        return None

    def _get_hook(self, name):
        def _hook(module, input, output):
            if self.hook_data.get(name) is None:
                self.hook_data[name] = []
            self.hook_data[name].append(output)
        return _hook

    def _register_hook(self, hook_name_layer):
        assert isinstance(hook_name_layer, dict)

        for hook_name, layer_name in hook_name_layer.items():
            layer = self._find_layer(self, layer_name)
            assert layer is not None, 'Hook layer is not found'
            handle = layer.register_forward_hook(self._get_hook(hook_name))
            self.handles.append(handle)
            print(f"layer {layer_name} is hooked")

    def _remove_hook_data(self):
        self.hook_data = dict()

    def forward_dummy(self, img):
        raise NotImplementedError('forward_dummy is not implemented')
