# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
import numpy as np
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results
from mmdet.do_something.scale_effect import ScaleEffect
from mmdet.models.utils.param_manager import ParamManager


def log_dist_per_scale(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    scale_effect = ScaleEffect(key='module.roi_head.bbox_roi_extractor')

    param_manager = ParamManager()
    param_manager.register_forward_hook(model, ['module.roi_head.bbox_roi_extractor'], type='input', detach=False)
    param_manager.register_forward_hook(model, ['module.roi_head.bbox_roi_extractor'], type='module', detach=False)
    param_manager.register_forward_hook(model, ['module.roi_head.bbox_roi_extractor'], type='output', detach=False)

    for i, data in enumerate(data_loader):
        # Convert gt_bboxes to proposals
        gt_bboxes = data['gt_bboxes'][0][0]
        ones = np.ones((len(gt_bboxes), 1))
        proposal = np.concatenate([gt_bboxes, ones], axis=1)
        proposal_list = [torch.tensor(proposal, dtype=torch.tensor(gt_bboxes).dtype)]

        # Forward
        data['img'] = [torch.concat([data['img'][0], data['img2'][0]], dim=0)]; del data['img2']
        data['gt_bboxes'] = [torch.concat([data['gt_bboxes'][0], data['gt_bboxes'][0]], dim=0)]
        data['gt_labels'] = [torch.concat([data['gt_labels'][0], data['gt_labels'][0]], dim=0)]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data, proposals=[proposal_list], num_views=2)

        # Compute scale and distance
        scale_effect._save_scale_and_dist(param_manager.hook_results, num_views=2, data=dict(gt_bboxes=[gt_bboxes.to('cuda'), gt_bboxes.to('cuda')]))

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    torch.save(scale_effect.scales, "/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.scales.pt")
    torch.save(scale_effect.dists, "/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dists.pt")

    _scales, _dists = scale_effect.scales, scale_effect.dists
    merged_scales, merged_dists = [], []
    for k, v in _scales.items():
        merged_scales += v
    for k, v in _dists.items():
        merged_dists += v

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.scatter([s.item() for s in merged_scales], [d.item() for d in merged_dists], s=0.5, marker='.')
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots.png')
        plt.close(fig)

        import pandas as pd
        df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
        df.sort_values(by='scale', inplace=True)

        df['group'] = pd.cut(df.scale, 20).cat.codes
        bins = df.groupby(df.group).mean()
        fig, ax = plt.subplots(1, 1)
        ax.scatter(df.scale, df.distance, s=0.5)
        plt.plot([s for s in bins.scale][1:-1], bins.distance[1:-1], 'r-', marker='*', lw=1)
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots_avg.png')
        plt.close(fig)

        # Filtered
        import pandas as pd
        df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
        # q_low = df['distance'].quantile(0.01)
        # q_high = df['distance'].quantile(0.99)
        # df_filted = df[(df['distance'] < q_high) & (df['distance'] > q_low)]
        df.sort_values(by='scale', inplace=True)

        df['group'] = pd.cut(df.scale, 20).cat.codes
        bins = df.groupby(df.group).mean()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.scatter([s for s in df.scale], [d for d in df.distance], s=0.5)
        plt.plot([s for s in bins.scale][1:-1], [d for d in bins.distance][1:-1], 'r-', marker='*', lw=1)
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots_s10.avg.png')
        plt.close(fig)

        # Filtered
        import pandas as pd
        df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
        df.sort_values(by='scale', inplace=True)
        df.distance = (df.distance - df.distance.mean()) / df.distance.std()  # Normalize

        df['group'] = pd.cut(df.scale, 20).cat.codes
        bins = df.groupby(df.group).mean()

        for s, d in zip(bins.scale, bins.distance):
            q_low = d.quantile(0.01)
            q_high = d.quantile(0.99)
            bins = bins[(d < q_high) & (d > q_low)]

        fig, ax = plt.subplots(1, 1)
        ax.scatter([s for s in df.scale], [d for d in df.distance], s=0.5)
        plt.plot([s for s in bins.scale][1:-1], [d for d in bins.distance][1:-1], 'r-', marker='*', lw=1)
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots_norm.avg.png')
        plt.close(fig)

        # Line
        import pandas as pd
        df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
        df.sort_values(by='scale', inplace=True)

        df['group'] = pd.cut(df.scale, 20).cat.codes
        fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.scatter([s for s in df.scale], [d for d in df.distance], s=0.5)
        m, b = np.polyfit([s for s in df.scale], [d for d in df.distance], 1)
        plt.plot([s for s in df.scale], [m * s + b for s in df.scale], 'r-', lw=2)
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots_line.png')
        plt.close(fig)

        # second-order
        import pandas as pd
        import seaborn as sns
        df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
        df.sort_values(by='scale', inplace=True)

        df['group'] = pd.cut(df.scale, 20).cat.codes
        fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.scatter([s for s in df.scale], [d for d in df.distance], s=0.5)
        sns.regplot([s for s in df.scale], [d for d in df.distance], order=2, scatter=False,
                    line_kws={'color': 'red', 'lw': 2})
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots_line2.png')
        plt.close(fig)

        # second-order only
        import pandas as pd
        import seaborn as sns
        df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
        df.sort_values(by='scale', inplace=True)

        df['group'] = pd.cut(df.scale, 20).cat.codes
        fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.regplot([s for s in df.scale], [d for d in df.distance], order=2, scatter=False,
                    line_kws={'color': 'red', 'lw': 2})
        plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.dots_only_line2.png')
        plt.close(fig)

    # second-order only
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({'scale': [s.item() for s in merged_scales], 'distance': [d.item() for d in merged_dists]})
    df.sort_values(by='scale', inplace=True)

    df['group'] = pd.cut(df.scale, 20).cat.codes
    fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.regplot([s for s in df.scale], [d for d in df.distance], order=2, scatter=False,
                line_kws={'color': 'red', 'lw': 2})
    ax.set_xlabel('Scale')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Brightness (dx=0.4)')
    plt.savefig('/ws/data/dshong/mmdetection/vars/scale_effect/brightness.dx4.scatter.png')
    plt.close(fig)

    return results
