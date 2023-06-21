# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
import numpy as np

import mmcv
from mmcv import Config

from mmdet.apis import init_random_seed, set_random_seed
from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.utils import (get_device, replace_cfg_vals, update_data_root)

type = 'none'
assert type in ['none', 'count_scale']


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)

    return args


def main():
    """ Load arguments and configs """
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    """ Setting """
    # set random seeds
    seed = init_random_seed(args.seed, device=get_device())
    set_random_seed(seed, deterministic=False)

    cfg.load_from = None
    cfg.log_config.hooks = [hook for (i, hook) in enumerate(cfg.log_config.hooks) if not hook.type in ['WandbLogger', 'MMDetWandbHook']]

    """ Load train data """
    train_loader_cfg = dict(
        samples_per_gpu=2, workers_per_gpu=0,
        num_gpus=1, dist=False, seed=seed,
        runner_type='EpochBasedRunner', persistent_workers=False)
    dataset = build_dataset(cfg.data.train)
    data_type = cfg.data.train.type.lower().replace('dataset', '')
    data_type = cfg.data.train.dataset.type.lower().replace('dataset', '') if data_type == 'repeat' else data_type
    dataloader = build_dataloader(dataset, **train_loader_cfg)

    """ Do something! """ # TODO
    sqrt_hw = []
    for batch in dataloader:
        if type == 'count_scale':
            # Compute sqrt(hw)
            gt_bboxes = batch['gt_bboxes'].data[0]
            for _gt_bboxes in gt_bboxes:
                for gt_bbox in _gt_bboxes:
                    sqrt_hw.append(torch.sqrt(
                        (gt_bbox[2]-gt_bbox[0]) * (gt_bbox[3]-gt_bbox[1])
                    ).item())
        else:
            img_metas = batch['img_metas']
            img = batch['img']
            gt_bboxes = batch['gt_bboxes']
            gt_labels = batch['gt_labels']
            break

    if type == 'count_scale':
        torch.save(sqrt_hw, osp.join(cfg.work_dir, f'{data_type}_sqrt_hw_train.pt'))

        import matplotlib.pyplot as plt
        for bin in [10, 20]:
            fig, ax = plt.subplots(1, 1)
            counts, edges, bars = ax.hist(sqrt_hw, bins=bin)
            total_num = np.sum(counts)
            ax.bar_label(bars, labels=[int(c/total_num*100) for c in counts])
            ax.set_ylabel('#objs')
            ax.set_xlabel('√HW')
            ax.set_title(f"{data_type}_train")
            plt.savefig(osp.join(cfg.work_dir, f'{data_type}_sqrt_hw_train_bin{bin}_p.png'))
            plt.close(fig)

if __name__ == '__main__':
    main()
