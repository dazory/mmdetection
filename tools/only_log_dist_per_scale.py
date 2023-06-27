# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import numpy as np
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import log_dist_per_scale
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--wandb', action='store_true', default=False, help='wandb mode')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    if args.debug:
        cfg.data.workers_per_gpu = 0
        test_dataloader_default_args['workers_per_gpu'] = 0
        cfg.load_from = None
        if not args.wandb:
            cfg.log_config.hooks = [hook for (i, hook) in enumerate(cfg.log_config.hooks) if
                                    not hook.type in ['WandbLogger', 'MMDetWandbHook', 'CustomMMDetWandbHook']]
        cfg.model.backbone.init_cfg = {}

    caterogy='brightness' # randcolor, brightness
    name = 'brightness.dx2' # brightness.dx4, brightness.dx2, randcolor

    scales = torch.load(f"/ws/data/dshong/mmdetection/vars/scale_effect/{caterogy}/{name}.scales.pt")
    dists = torch.load(f"/ws/data/dshong/mmdetection/vars/scale_effect/{caterogy}/{name}.dists.pt")
    entropy = torch.load(f"/ws/data/dshong/mmdetection/vars/scale_effect/{caterogy}/{name}.entropy.pt")

    merged_scales, merged_dists, merged_entropy = [], [], []
    for k, v in scales.items(): merged_scales += v
    for k, v in dists.items(): merged_dists += v
    for k, v in entropy.items(): merged_entropy += v

    def ridge_plot(scales, distances, name, hspace=-0.9):
        import pandas as pd
        import numpy as np
        from sklearn.neighbors import KernelDensity
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as grid_spec

        num_bins = 10

        df = pd.DataFrame({'scale': [s.item() for s in scales], 'distance': [d.item() for d in distances]})
        df.sort_values(by='scale', inplace=True)
        df['group'] = pd.cut(df.scale, num_bins).cat.codes
        _num_samples = len(df)
        # _df_in = df.sort_values(by='distance')['distance'][0:-int(_num_samples*0.01)]
        if 'brightness' in name:
            if '2' in name:
                x_min, x_max = 0, 10000  # _df_in.min(), _df_in.max()
            elif '4' in name:
                x_min, x_max = 500, 16000
            elif '6' in name:
                x_min, x_max = 2000, 20000  # _df_in.min(), _df_in.max()
        else:
            x_min, x_max = 2000, 18000

        gs = grid_spec.GridSpec(num_bins, 1, hspace=hspace)
        fig = plt.figure(figsize=(10, 10))
        ax_objs = []
        max_coords = []
        for i in range(num_bins):
            # Create new axes object and appending to ax_objs
            ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
            if i == 0:
                ax_objs[-1].set_title(f'{name}')

            # Plot the distribution
            cmap = plt.cm.magma
            plot = df[df['group'] == i].distance.plot(kind='kde', ax=ax_objs[-1], legend=False, title='', color=cmap(i/num_bins))

            # Grab x and y data from the kde plot
            x = plot.get_children()[0]._x
            y = plot.get_children()[0]._y
            max_coords.append((x[np.argmax(y)], y[np.argmax(y)]))
            ax_objs[-1].plot([x[np.argmax(y)]], [y[np.argmax(y)]], 'o', color='red')

            # Fill the space beneath the distribution
            ax_objs[-1].fill_between(x, y, color=cmap(i / num_bins))

            # Set uniform x and y lims
            ax_objs[-1].set_xlim(x_min, x_max)
            ax_objs[-1].set_ylim(0, 0.0005)

            # Make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)

            # Remove borders, axis ticks, and labels
            ax_objs[-1].set_yticks([])
            ax_objs[-1].set_ylabel('')
            if i == num_bins - 1:
                pass
            else:
                ax_objs[-1].set_xticklabels([])

            for s in ['top', 'right', 'left', 'bottom']:
                ax_objs[-1].spines[s].set_visible(False)

            scale = df[df['group'] == i].scale
            scale = f"({int(scale.min())}, {int(scale.max())})"
            ax_objs[-1].text(-0.02, 0.05, scale, horizontalalignment='center', verticalalignment='center', transform=ax_objs[-1].transAxes)

        plt.tight_layout()
        plt.savefig(f"/ws/data/dshong/mmdetection/vars/scale_effect2/2_{name}_ridge.png")

    ridge_plot(merged_scales, merged_dists, name)

if __name__ == '__main__':
    main()
