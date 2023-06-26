# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import torch
import numpy as np
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results
from mmdet.do_something.scale_effect import ScaleEffect
from mmdet.models.utils.param_manager import ParamManager
from mmdet.core import bbox_overlaps


def log_dist_per_scale(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3):
    model.eval()
    model2 = copy.deepcopy(model).eval()
    results1, results2 = [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    scale_effect = ScaleEffect(key='module.roi_head.bbox_roi_extractor')

    param_manager = ParamManager()
    param_manager.register_forward_hook(model, ['module.roi_head.bbox_roi_extractor'], type='input', detach=False)
    param_manager.register_forward_hook(model, ['module.roi_head.bbox_roi_extractor'], type='module', detach=False)
    param_manager.register_forward_hook(model, ['module.roi_head.bbox_roi_extractor'], type='output', detach=False)
    param_manager2 = ParamManager()
    param_manager2.register_forward_hook(model2, ['module.roi_head.bbox_roi_extractor'], type='input', detach=False)
    param_manager2.register_forward_hook(model2, ['module.roi_head.bbox_roi_extractor'], type='module', detach=False)
    param_manager2.register_forward_hook(model2, ['module.roi_head.bbox_roi_extractor'], type='output', detach=False)

    for i, data in enumerate(data_loader):
        # Convert gt_bboxes to proposals
        gt_bboxes = data['gt_bboxes'][0][0]
        ones = np.ones((len(gt_bboxes), 1))
        proposal = np.concatenate([gt_bboxes, ones], axis=1)
        proposal_list = [torch.tensor(proposal, dtype=torch.tensor(gt_bboxes).dtype)]

        # Forward
        # data['img'] = [torch.concat([data['img'][0], data['img2'][0]], dim=0)]; del data['img2']
        # data['gt_bboxes'] = [torch.concat([data['gt_bboxes'][0], data['gt_bboxes'][0]], dim=0)]
        # data['gt_labels'] = [torch.concat([data['gt_labels'][0], data['gt_labels'][0]], dim=0)]
        with torch.no_grad():
            result1 = model(return_loss=False, rescale=True, **data, proposals=[proposal_list], num_views=1)
            data['img'] = data['img2']
            del data['img2']
            result2 = model2(return_loss=False, rescale=True, **data, proposals=[proposal_list], num_views=1)

        _result1 = torch.tensor(np.concatenate(result1[0], axis=0))
        _result2 = torch.tensor(np.concatenate(result2[0], axis=0))
        if len(_result1) < len(_result2):
            _result1 = torch.cat([_result1, torch.zeros(len(_result2) - len(_result1), _result1.shape[1])])
            axis=0
        else:
            _result2 = torch.cat([_result2, torch.zeros(len(_result1) - len(_result2), _result2.shape[1])])
            axis=1

        # Compute scale and distance
        scale_effect._save_scale_and_dist(param_manager.hook_results, hook_results2=param_manager2.hook_results, num_views=2,
                                          data=dict(gt_bboxes=[gt_bboxes.to('cuda'), gt_bboxes.to('cuda')]),
                                          roi_extractor=model.module.roi_head.bbox_roi_extractor)

        results1.extend(result1)
        results2.extend(result2)
        batch_size = len(result1)
        for _ in range(batch_size):
            prog_bar.update()

    if False:
        dx = 0.6
        name = 'brightness.dx6'

        torch.save(scale_effect.scales, f"/ws/data/dshong/mmdetection/vars/scale_effect/{name}.scales.pt")
        torch.save(scale_effect.dists, f"/ws/data/dshong/mmdetection/vars/scale_effect/{name}.dists.pt")
        torch.save(scale_effect.entropy, f"/ws/data/dshong/mmdetection/vars/scale_effect/{name}.entropy.pt")
        torch.save(scale_effect.jsd, f"/ws/data/dshong/mmdetection/vars/scale_effect/{name}.jsd.pt")

        _scales, _dists, _entropy, _jsd = scale_effect.scales, scale_effect.dists, scale_effect.entropy, scale_effect.jsd
        merged_scales, merged_dists, merged_entropy, merged_jsd = [], [], [], []
        for k, v in _scales.items():
            merged_scales += v
        for k, v in _dists.items():
            merged_dists += v
        for k, v in _entropy.items():
            merged_entropy += v
        for k, v in _jsd.items():
            merged_jsd += v

        # second-order only
        def vis(scales, distances, is_jsd=False, dx=-1.0, name='brightness', color=None,
                log_scale=False, sqr_scale=False, show_legend=False, scatter=False, order=1, label="", save=True,
                fig=None, ax=None,
                remove_outliers=False, color2='red', colorbar=True):
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            if log_scale:
                _distance = [np.log(d.item()) for d in distances]
            elif sqr_scale:
                _distance = [d.item() ** 2 for d in distances]
            else:
                _distance = [d.item() for d in distances]

            if color is None:
                df = pd.DataFrame({'scale': [s.item() for s in scales], 'distance': _distance})
            else:
                color = torch.tensor([c.item() for c in color])
                color = (color - min(color)) / (max(color) - min(color))
                df = pd.DataFrame({'scale': [s.item() for s in scales], 'distance': _distance,
                                   'color': [c.item() for c in list(color)]})
            df.sort_values(by='scale', inplace=True)
            if remove_outliers:
                num_samples = 100
                df.scale = df.scale[num_samples:-num_samples]
                df.distance = df.distance[num_samples:-num_samples]
                df.color = df.color[num_samples:-num_samples]

            df['group'] = pd.cut(df.scale, 20).cat.codes

            if ax is None:
                fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            if scatter:
                sc = ax.scatter([s for s in df.scale], [d for d in df.distance], s=0.5, c=[c for c in df.color])
                if colorbar:
                    plt.colorbar(sc)
            if order > 0:
                sns.regplot([s for s in df.scale], [d for d in df.distance], order=order, scatter=False,
                            line_kws={'color': color2, 'lw': 2}, label=label)

            ax.set_xlabel('Scale')
            ax.set_ylabel('Euclidean Distance' if not is_jsd else 'JSD')
            ax.set_title(f'{name}{f" (dx={dx:.1f})" if dx > 0 else ""}')
            if show_legend:
                ax.legend()
            if save:
                plt.savefig(
                    f'/ws/data/dshong/mmdetection/vars/scale_effect/{name}.{"jsd_" if is_jsd else ""}{"scatter" if scatter else "reg"}{".log" if log_scale else ""}{".sqr" if sqr_scale else ""}{".inlier" if remove_outliers else ""}{order if order > 0 else ""}.png')
                plt.close(fig)
            else:
                return fig, ax

        for i in range(1, 10):
            vis(merged_scales, merged_dists, dx=dx, name=name, scatter=True, color=merged_entropy, order=i)
            vis(merged_scales, merged_jsd, is_jsd=True, dx=dx, name=name, scatter=True, color=merged_entropy, order=i)
            vis(merged_scales, merged_dists, dx=dx, name=name, scatter=False, order=i)
            vis(merged_scales, merged_jsd, is_jsd=True, dx=dx, name=name, scatter=False, color=merged_entropy, order=i)

    if False:
        scale_dx2 = torch.load("/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx2.scales.pt")
        dist_dx2 = torch.load("/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx2.dists.pt")
        entropy_dx2 = torch.load("/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx2.entropy.pt")
        merged_scales_dx2, merged_dists_dx2, merged_entropy_dx2 = [], [], []
        for k, v in scale_dx2.items():
            merged_scales_dx2 += v
        for k, v in dist_dx2.items():
            merged_dists_dx2 += v
        for k, v in entropy_dx2.items():
            merged_entropy_dx2 += v

        scale_dx4 = torch.load("/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.scales.pt")
        dist_dx4 = torch.load("/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dists.pt")
        entropy_dx4 = torch.load("/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.entropy.pt")
        merged_scales_dx4, merged_dists_dx4, merged_entropy_dx4 = [], [], []
        for k, v in scale_dx4.items():
            merged_scales_dx4 += v
        for k, v in dist_dx4.items():
            merged_dists_dx4 += v
        for k, v in entropy_dx4.items():
            merged_entropy_dx4 += v

        for i in range(1, 7):
            # fig, ax = vis(merged_scales_dx2, merged_dists_dx2, dx=0.2, label='dx=0.2', scatter=True, color=merged_entropy_dx2, color2='red', order=i, log_scale=True, name="", save=False, colorbar=False)
            # vis(merged_scales_dx4, merged_dists_dx4, fig=fig, ax=ax, dx=0.2, label='dx=0.4', scatter=True, color=merged_entropy_dx4, color2='blue', order=i, log_scale=True, name='brightness.dx2.dx4', save=True)
            fig, ax = vis(merged_scales_dx2, merged_dists_dx2, dx=0.2, label='dx=0.2', scatter=False, color=merged_entropy_dx2, color2='red', order=i, log_scale=True, name="", save=False, colorbar=False)
            vis(merged_scales_dx4, merged_dists_dx4, fig=fig, ax=ax, dx=0.2, label='dx=0.4', scatter=False, color=merged_entropy_dx4, color2='blue', order=i, log_scale=True, name='brightness.dx2.dx4', save=True)

    return results1, results2
