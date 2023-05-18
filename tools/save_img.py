'''
how to use?
1. EDSR
    `python3 -u /ws/external/tools/save_img.py /ws/external/configs/_dshong/deepaugment/city_deepaugment_EDSR.py`
2. CAE
    `python3 -u /ws/external/tools/save_img.py /ws/external/configs/_dshong/deepaugment/city_deepaugment_CAE.py`
You can see the saved images at `/ws/data/cityscapes_EDSR` or `/ws/data/cityscapes_CAE`.
'''

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
import gc
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist)

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes, update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

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

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
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

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    for i, t in enumerate(data_loader.dataset.pipeline.transforms):
        if t.__class__.__name__ in ['Resize', 'RandomFlip']:
            data_loader.dataset.pipeline.transforms.remove(t)
        elif t.__class__.__name__ in ['Collect']:
            meta_keys = ('filename', 'ori_filename', 'ori_shape',
                         'img_shape', 'pad_shape', 'img_norm_cfg')
            data_loader.dataset.pipeline.transforms[i].meta_keys = meta_keys

    if not distributed:
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                gc.collect()
                torch.cuda.empty_cache()
                batch_size = len(data['img'])
                for _ in range(batch_size):
                    prog_bar.update()
    else:
        raise NotImplementedError
        dataset = data_loader.dataset
        for t in dataset.pipeline.transforms:
            if t.__class__.__name__ == 'DeepAugment':
                deepaugment = t
        deepaugment.net = build_ddp(
            deepaugment.net,
            cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                gc.collect()
                torch.cuda.empty_cache()
                batch_size = len(data['img'])
                for _ in range(batch_size):
                    prog_bar.update()

    print(f"Done!")

if __name__ == '__main__':
    main()
