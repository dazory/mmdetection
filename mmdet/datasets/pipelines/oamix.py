# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch

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
    dict(type='BboxRotate', level=DEFAULT_LEVEL, randomness=not RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxShearX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxShearY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxTranslateX', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
    dict(type='BboxTranslateY', level=DEFAULT_LEVEL, randomness=RANDOMNESS, blur=DEFUALT_BLUR),
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
        self.aug_list = self._get_augs(version)
        self.num_augs = len(self.aug_list)
        self.mixture_width = mixture_width  # The number of operations to be mixed
        self.mixture_depth = mixture_depth

        self.aug_prob_coeff = 1.0
        self.mixing_weights = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.aug_prob_coeff] * self.mixture_width))
        self.sample_weights = torch.distributions.beta.Beta(torch.tensor([self.aug_prob_coeff]), torch.tensor([self.aug_prob_coeff]))

    def __call__(self, data):
        """
        Args:
            imgs (torch.Tensor): Image tensors with shape (bs, c, h, w).
        """
        imgs = data['img']
        gt_bboxes = data['gt_bboxes']
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
        assert len(imgs.shape) == 4, f"imgs shape should be (b, c, h, w), but got {imgs.shape}"
        imgs /= 255.0
        bs = imgs.shape[0]
        device = imgs.get_device()

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

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'

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
        elif version == '1.0':
            aug_cfg_list = ALL_COLOR_AUGS + ALL_BBOX_SPATIAL_AUGS_WITH_BLUR + []
        else:
            raise NotImplementedError(f'Not support OA-Mix version {version}')

        aug_list = []
        for aug_cfg in aug_cfg_list:
            aug_list.append(build_from_cfg(aug_cfg, TRANSFORMATIONS))
        return aug_list
