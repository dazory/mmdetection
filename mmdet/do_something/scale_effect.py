import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core import bbox2roi

from .builder import DO_SOMETHING


@DO_SOMETHING.register_module()
class ScaleEffect:
    def __init__(self, key='roi_head.bbox_roi_extractor'):
        super().__init__()
        scales = dict(s=[], m=[], l=[])
        dists = dict(s=[], m=[], l=[])
        entropy = dict(s=[], m=[], l=[])
        jsd = dict(s=[], m=[], l=[])
        self.storage = dict(scales=scales, dists=dists, entropy=entropy, jsd=jsd)
        self.outputs = dict()
        self.ths = [32, 96]
        self.min_samples = 10
        self.key = key

    def _save_scale_and_dist(self, hook_results, num_views, data, hook_results2=None, **kwargs):
        assert num_views == 2, "Only support 2 views for now."
        outputs = dict()

        # Prepare data
        feats, _ = hook_results['input'][self.key]
        module = hook_results['module'][self.key]
        if hook_results2 is not None:
            feats2, _ = hook_results2['input'][self.key]
            feats = tuple([torch.cat([feats[i], feats2[i]]) for i in range(len(feats))])

        gt_rois = bbox2roi(data['gt_bboxes'])
        gt_bbox_feats = module(feats[:module.num_inputs], gt_rois)
        gt_bbox_feats = gt_bbox_feats.reshape(len(gt_bbox_feats), -1)
        div = len(gt_bbox_feats) // num_views

        gt_bbox_probs = F.softmax(gt_bbox_feats, dim=1)
        log_gt_bbox_probs = torch.log(gt_bbox_probs + 1e-6)
        entropy = -torch.sum(gt_bbox_probs * log_gt_bbox_probs, dim=1)
        mean_entropy = (entropy[:div] + entropy[div:]) / 2.
        outputs.update({'entropy': mean_entropy})

        # Compute Jensen-Shannon divergence between features
        p_mixture = torch.clamp((gt_bbox_probs[:div] + gt_bbox_probs[div:]) / 2.0, 1e-6, 1.).log()
        jsd = (F.kl_div(p_mixture, gt_bbox_probs[:div], reduction='none') +
                F.kl_div(p_mixture, gt_bbox_probs[div:], reduction='none')) / 2.
        jsd = torch.sum(jsd, dim=1)
        outputs.update({'jsd': jsd})

        # Compute Euclidean distance between features
        euclidean_dists = F.pairwise_distance(
            gt_bbox_feats[:div], gt_bbox_feats[div:], p=1)  # (1024,)
        outputs.update({'dists': euclidean_dists})

        # Compute scale
        sqrt_hws = torch.sqrt(
            (gt_rois[:, 3] - gt_rois[:, 1]) * (gt_rois[:, 4] - gt_rois[:, 2])
        )
        sqrt_hws = torch.chunk(sqrt_hws, num_views)[0]
        outputs.update({'scales': sqrt_hws})

        idx_s = sqrt_hws < self.ths[0]
        idx_m = (self.ths[0] <= sqrt_hws) * (sqrt_hws < self.ths[1])
        idx_l = self.ths[1] <= sqrt_hws

        for key in ['scales', 'dists', 'entropy', 'jsd']:
            for (_scale, _idx) in [('s', idx_s), ('m', idx_m), ('l', idx_l)]:
                self.storage[key][_scale] += list(outputs[key][_idx])

    def __call__(self, hook_results, num_views, outputs, **kwargs):
        assert num_views == 2, "Only support 2 views for now."
        self._save_scale_and_dist(hook_results, num_views, **kwargs)
        for key, data in self.storage.items():
            for _scale, _value in data.items():
                if len(_value) > self.min_samples:
                    self.outputs[f"{key}_{_scale}"] = torch.mean(torch.stack(_value[:self.min_samples]))
                    del _value[:self.min_samples]

        outputs.update(self.outputs)

        return outputs