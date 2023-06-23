import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core import bbox2roi

from .builder import DO_SOMETHING


@DO_SOMETHING.register_module()
class ScaleEffect:
    def __init__(self, key='roi_head.bbox_roi_extractor'):
        super().__init__()
        self.scales = dict(s=[], m=[], l=[])
        self.dists = dict(s=[], m=[], l=[])
        self.outputs = dict()
        self.ths = [32, 96]
        self.min_samples = 10
        self.key = key

    def _save_scale_and_dist(self, hook_results, num_views, data, **kwargs):
        assert num_views == 2, "Only support 2 views for now."

        # Prepare data
        feats, _ = hook_results['input'][self.key]
        module = hook_results['module'][self.key]

        gt_rois = bbox2roi(data['gt_bboxes'])
        gt_bbox_feats = module(feats[:module.num_inputs], gt_rois)
        gt_bbox_feats = gt_bbox_feats.reshape(len(gt_bbox_feats), -1)
        div = len(gt_bbox_feats) // num_views

        # Compute Euclidean distance between features
        euclidean_dists = F.pairwise_distance(
            gt_bbox_feats[:div], gt_bbox_feats[div:], p=1)  # (1024,)

        # Compute scale
        sqrt_hws = torch.sqrt(
            (gt_rois[:, 3] - gt_rois[:, 1]) * (gt_rois[:, 4] - gt_rois[:, 2])
        )
        sqrt_hws = torch.chunk(sqrt_hws, num_views)[0]

        idx_s = sqrt_hws < self.ths[0]
        idx_m = (self.ths[0] <= sqrt_hws) * (sqrt_hws < self.ths[1])
        idx_l = self.ths[1] <= sqrt_hws

        self.scales['s'] += list(sqrt_hws[idx_s])
        self.dists['s'] += list(euclidean_dists[idx_s])
        self.scales['m'] += list(sqrt_hws[idx_m])
        self.dists['m'] += list(euclidean_dists[idx_m])
        self.scales['l'] += list(sqrt_hws[idx_l])
        self.dists['l'] += list(euclidean_dists[idx_l])

    def __call__(self, hook_results, num_views, outputs, **kwargs):
        assert num_views == 2, "Only support 2 views for now."
        self._save_scale_and_dist(hook_results, num_views, **kwargs)
        for k, v in self.scales.items():
            if len(v) > self.min_samples:
                self.outputs[f"scale_{k}"] = torch.mean(torch.stack(v[:self.min_samples]))
                del v[:self.min_samples]

        for k, v in self.dists.items():
            if len(v) > self.min_samples:
                self.outputs[f"dist_{k}"] = torch.mean(torch.stack(v[:self.min_samples]))
                del v[:self.min_samples]

        outputs.update(self.outputs)

        return outputs