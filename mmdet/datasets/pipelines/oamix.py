# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch

from mmcv.utils import build_from_cfg
from ..basic_transforms.color_ops import *
from ..builder import PIPELINES, TRANSFORMATIONS


ALL_COLOR_AUGS = [
    dict(type='AutoContrast', p=1.0),
    dict(type='Equalize', p=1.0),
    dict(type='Posterize', level=4, p=1.0),
    dict(type='Solarize', level=4, p=1.0),
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

    def __call__(self, imgs):
        """
        Args:
            imgs (torch.Tensor): Image tensors with shape (bs, c, h, w).
        """
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
                imgs[aug_mask[j] == 1] = aug(copy.deepcopy(imgs))[aug_mask[j] == 1]

            # Mix imgs
            mixed_imgs += mixing_weights[:, i] * imgs  # (bs, ) * (bs, c, h, w)

        augmixed_imgs = (1 - sample_weights) * imgs + sample_weights * mixed_imgs
        return augmixed_imgs

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'

    def _get_augs(self, version):
        if version == '0.0':
            aug_cfg_list = ALL_COLOR_AUGS + []
        else:
            raise NotImplementedError(f'Not support OA-Mix version {version}')

        aug_list = []
        for aug_cfg in aug_cfg_list:
            aug_list.append(build_from_cfg(aug_cfg, TRANSFORMATIONS))
        return aug_list
