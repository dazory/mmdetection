# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mmcv.utils import build_from_cfg
from ..basic_transforms.color_ops import *
from ..builder import PIPELINES, TRANSFORMATIONS

DEFAULT_PROB = 1.0
DEFAULT_LEVEL = 3
DEFUALT_BLUR = dict(Kx=101, Ky=101, r=1/4)
RANDOMNESS = True

ALL_COLOR_AUGS = [
    dict(type='AutoContrast', p=DEFAULT_PROB),
    dict(type='Equalize', p=DEFAULT_PROB),
    dict(type='Posterize', level=DEFAULT_LEVEL, p=DEFAULT_PROB, randomness=RANDOMNESS),
    dict(type='Solarize', level=DEFAULT_LEVEL, p=DEFAULT_PROB, randomness=RANDOMNESS),
]

ALL_SPATIAL_AUGS = [
    dict(type='Rotate', level=DEFAULT_LEVEL, padding_mode='zeros', randomness=RANDOMNESS),
    dict(type='ShearX', level=DEFAULT_LEVEL, padding_mode='zeros', randomness=RANDOMNESS),
    dict(type='ShearY', level=DEFAULT_LEVEL, padding_mode='zeros', randomness=RANDOMNESS),
    dict(type='TranslateX', level=DEFAULT_LEVEL, padding_mode='zeros', randomness=RANDOMNESS),
    dict(type='TranslateY', level=DEFAULT_LEVEL, padding_mode='zeros', randomness=RANDOMNESS),
]

ALL_BBOX_SPATIAL_AUGS = [
    dict(type='BboxRotate', level=DEFAULT_LEVEL, randomness=RANDOMNESS),
    dict(type='BboxShearX', level=DEFAULT_LEVEL, randomness=RANDOMNESS),
    dict(type='BboxShearY', level=DEFAULT_LEVEL, randomness=RANDOMNESS),
    dict(type='BboxTranslateX', level=DEFAULT_LEVEL, randomness=RANDOMNESS),
    dict(type='BboxTranslateY', level=DEFAULT_LEVEL, randomness=RANDOMNESS),
]

ALL_BBOX_SPATIAL_AUGS_WITH_BLUR = [
    dict(type='BboxRotate', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxShearX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxShearY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxTranslateX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxTranslateY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
]

ALL_BG_SPATIAL_AUGS_WITH_BLUR = [
    dict(type='NotBboxRotate', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='NotBboxShearX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='NotBboxShearY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='NotBboxTranslateX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='NotBboxTranslateY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
]

DEFAULT_RAND_BG_CFG = dict(rand_type='bg', max_num_bboxes=(3, 10), scales=(0.01, 0.2), ratios=(0.3, 1 / 0.3))
DEFAULT_RAND_FG_CFG = dict(rand_type='fg', max_num_bboxes=(3, 10), scales=(0.001, 0.01), ratios=(0.3, 1 / 0.3))
ALL_RAND_BG_BBOX_SPATIAL_AUGS_WITH_BLUR = [
    dict(type='RandBboxRotate', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_BG_CFG),
    dict(type='RandBboxShearX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_BG_CFG),
    dict(type='RandBboxShearY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_BG_CFG),
    dict(type='RandBboxTranslateX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_BG_CFG),
    dict(type='RandBboxTranslateY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_BG_CFG),
]

ALL_RAND_FG_BBOX_SPATIAL_AUGS_WITH_BLUR = [
    dict(type='RandBboxRotate', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_FG_CFG),
    dict(type='RandBboxShearX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_FG_CFG),
    dict(type='RandBboxShearY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_FG_CFG),
    dict(type='RandBboxTranslateX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_FG_CFG),
    dict(type='RandBboxTranslateY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR, rand_cfg=DEFAULT_RAND_FG_CFG),
]


@PIPELINES.register_module()
class OAMix:
    """OA-Mix.

    This data augmentation is proposed in `Object-Aware
    Domain Generalization for Object Detection <https://TODO>`_.

    Args:

    Examples:

    """

    def __init__(self, version,
                 mixture_width=3,
                 mixture_depth=-1):
        self.version = version
        self.aug_list = self._get_augs(version)
        self.num_augs = len(self.aug_list)
        self.mixture_width = mixture_width  # The number of operations to be mixed
        self.mixture_depth = mixture_depth

        self.aug_prob_coeff = 1.0
        self.mixing_weights = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.aug_prob_coeff] * self.mixture_width))
        self.sample_weights = torch.distributions.beta.Beta(torch.tensor([self.aug_prob_coeff]), torch.tensor([self.aug_prob_coeff]))

        self.mean_overlap = 0.0

        # visualize_proposals
        self.vis_interval = 100
        self.save_path = '/ws/data/dshong/mmdetection/oamix/visualize_proposals'
        self.save_i = 0

    @staticmethod
    def _measure_overlap(gt_bboxes, proposal_list, img_shape):
        # img_shape = (h, w)
        batch_size = len(gt_bboxes)
        mean_overlap = 0.0
        bbox_mask = torch.zeros(img_shape, dtype=torch.uint8)
        for i in range(batch_size):
            for bbox in gt_bboxes[i]:
                bbox_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

            overlaps = 0.0
            for proposal in proposal_list[i][:, :-1]:
                proposal_mask = torch.zeros(img_shape, dtype=torch.uint8)
                (x1, y1, x2, y2) = proposal
                proposal_mask[int(y1):int(y2), int(x1):int(x2)] = 1

                overlap_mask = bbox_mask & proposal_mask
                overlap = overlap_mask.sum() / ((x2 - x1) * (y2 - y1))
                overlaps += overlap

            mean_overlap += overlaps / len(proposal_list[0])
        mean_overlap /= batch_size
        return mean_overlap

    def __call__(self, data):
        """
        Args:
            imgs (torch.Tensor): Image tensors with shape (bs, c, h, w).
        """
        imgs = data['img']
        gt_bboxes = data['gt_bboxes']; proposal_list = data.get('proposal_list', None);
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
        assert len(imgs.shape) == 4, f"imgs shape should be (b, c, h, w), but got {imgs.shape}"
        imgs /= 255.0
        bs = imgs.shape[0]
        device = imgs.get_device()

        self.mean_overlap = self._measure_overlap(gt_bboxes, proposal_list, img_shape=(imgs.shape[-2], imgs.shape[-1])).item()

        # visualize proposals
        if self.save_i % self.vis_interval == 0:
            self._visualize_proposals(imgs, proposal_list, f"{self.save_path}/{self.save_i}.png")

        mixing_weights = torch.zeros((bs, self.mixture_width), device=device)
        sample_weights = torch.zeros((bs,), device=device)
        for i in range(bs):
            mixing_weights[i] = self.mixing_weights.sample().type(torch.float32)
            sample_weights[i] = self.sample_weights.sample().type(torch.float32)

        mixed_imgs = torch.zeros_like(imgs)
        for i in range(self.mixture_width):
            aug_imgs = copy.deepcopy(imgs)
            depth = self.mixture_depth if self.mixture_depth > 0 \
                else np.random.randint(1, 4)

            # Generate aug mask for parallel aug
            # Select mixture_width number of augs from aug_list for each img in batch
            aug_mask = torch.zeros((self.num_augs, bs))  # (depth, bs)
            for j in range(bs):
                aug_mask[torch.randperm(self.num_augs)[:depth], j] = 1

            # Remove augs that are not applied to any img in batch
            aug_list = copy.deepcopy(self.aug_list)
            ignored_inds = torch.sum(aug_mask, dim=1) == 0
            aug_list = [aug for i, aug in enumerate(aug_list) if not ignored_inds[i]]
            aug_mask = aug_mask[~ignored_inds, :]

            # only apply aug to imgs where aug_mask[i] == 1
            for j, aug in enumerate(aug_list):
                aug_imgs[aug_mask[j] == 1] = \
                    aug(aug_imgs, bboxes=gt_bboxes)[aug_mask[j] == 1]

            # Mix imgs
            mixed_imgs += mixing_weights[:, i] * aug_imgs  # (bs, ) * (bs, c, h, w)

        augmented_img = (1 - sample_weights) * imgs + sample_weights * mixed_imgs
        data['img'] = augmented_img
        return data

    def _visualize_proposals(self, imgs, proposal_list, save_path):
        fig, ax = plt.subplots(1, 1)
        img_vis = imgs[0].permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(img_vis)
        proposals = proposal_list[0]
        for i in range(len(proposals)):
            bbox = proposals[i].cpu().detach().numpy()
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            patch = Rectangle(xy=(x1, y1), width=(x2 - x1), height=(y2 - y1),
                              linewidth=1, edgecolor='none', facecolor='#e41a1c', alpha=bbox[4].item())
            ax.add_patch(patch)
        plt.savefig(f'{save_path}')
        plt.close(fig)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.version})'

    def _get_augs(self, version):
        if version == '0.0':
            aug_cfg_list = ALL_COLOR_AUGS + []
        elif version == '0.1':
            aug_cfg_list = ALL_SPATIAL_AUGS + []
        elif version == '0.2':
            aug_cfg_list = ALL_BBOX_SPATIAL_AUGS + []
        elif version == '0.3':
            aug_cfg_list = ALL_COLOR_AUGS + ALL_SPATIAL_AUGS + ALL_BBOX_SPATIAL_AUGS
        elif version == '0.4':
            aug_cfg_list = ALL_BBOX_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '1.0':  # fg
            aug_cfg_list = ALL_COLOR_AUGS + ALL_BBOX_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '1.1':  # bg
            aug_cfg_list = ALL_COLOR_AUGS + ALL_BG_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '1.2':  # fg + bg
            aug_cfg_list = ALL_COLOR_AUGS + ALL_BBOX_SPATIAL_AUGS_WITH_BLUR + ALL_BG_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '1.3.1':  # rand_bg
            aug_cfg_list = ALL_COLOR_AUGS + ALL_RAND_BG_BBOX_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '1.3.2':  # rand_fg
            aug_cfg_list = ALL_COLOR_AUGS + ALL_RAND_FG_BBOX_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '1.3.3':  # rand_fg + rand_bg
            aug_cfg_list = ALL_COLOR_AUGS + ALL_RAND_BG_BBOX_SPATIAL_AUGS_WITH_BLUR + ALL_RAND_FG_BBOX_SPATIAL_AUGS_WITH_BLUR + []
        elif version == '2.0':  # proposals
            raise NotImplementedError('Not support OA-Mix version 2.0')  # TODO: use proposals
            aug_cfg_list = [dict(type='BboxRotate', level=DEFAULT_LEVEL, randomness=not RANDOMNESS, blur=DEFUALT_BLUR)]
        else:
            raise NotImplementedError(f'Not support OA-Mix version {version}')

        aug_list = []
        for aug_cfg in aug_cfg_list:
            aug_list.append(build_from_cfg(aug_cfg, TRANSFORMATIONS))
        return aug_list
