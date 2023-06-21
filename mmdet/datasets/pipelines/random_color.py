# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
import math
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomColor:
    def __init__(self,
                 num_views=1,
                 cut_max=200,
                 T_min=0.0, T_max=0.05,
                 keep_orig=True,
                 ):
        super().__init__()
        self.num_views = num_views
        self.cut_max = cut_max
        self.T_min, self.T_max = T_min, T_max
        self.keep_orig = keep_orig
        if self.keep_orig and self.num_views < 2: warnings.warn(
            f'keep_orig==True but num_views is less than two. This has no effect.')

    def aug(self, img):
        assert np.max(img) <= 1 and 0 <=np.min(img), "img must be in [0, 1]"

        # Sample parameters
        cut = np.random.randint(1, self.cut_max)
        k_min = np.random.randint(1, cut)

        # Augmentation
        freqs = np.zeros(img.shape, dtype=np.float32)
        for i in range(k_min, min(k_min+20, cut)):
            beta = np.random.rand(3) * np.sqrt(np.random.uniform(self.T_min, self.T_max))
            freqs += beta * np.sin(math.pi * i * copy.deepcopy(img))
        img_aug = np.clip(img + freqs, 0, 1)

        return img_aug

    def __call__(self, results):
        img = results['img']
        ori_type = img.dtype
        if img.dtype == np.uint8:
            img = img / 255

        results['custom_fields'] = []

        i_min = 2 if self.keep_orig else 1
        for i in range(i_min, self.num_views + 1):
            img_aug = self.aug(copy.deepcopy(img))
            dx = np.mean(np.abs(img - img_aug))

            if ori_type == np.uint8:
                img_aug = (img_aug * 255).astype(ori_type)

            if i > 1:
                results[f'img{i}'] = img_aug
                results[f'dx{i}'] = dx
                results['custom_fields'] += [f'img{i}', f'dx{i}']
                results['img_fields'] += [f'img{i}']
            else:
                results[f'img'] = img_aug
                results[f'dx'] = dx
                results['custom_fields'] += ['dx']

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dx={self.dx}, num_views={self.num_views}, cut_max={self.cut_max}, T in [{self.T_min:.3f}, {self.T_max:.3f}] )'
        return repr_str
