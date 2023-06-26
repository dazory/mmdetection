# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
import math
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Brightness:
    def __init__(self,
                 dx,
                 num_views=1,
                 err_max=0.05,
                 keep_orig=True,
                 ):
        super().__init__()
        self.dx = dx
        self.num_views = num_views
        self.err_max = err_max
        self.keep_orig = keep_orig
        if self.keep_orig and self.num_views < 2: warnings.warn(
            f'keep_orig==True but num_views is less than two. This has no effect.')

    def aug(self, img):
        assert np.max(img) <= 1 and 0 <=np.min(img), "img must be in [0, 1]"

        dx = self.dx
        if np.random.rand() < 0.5:
            dx *= -1
        img_aug = np.clip(img + dx, 0, 1)
        _dx = np.mean(np.abs(img - img_aug))

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
        repr_str += f'(dx={self.dx}, num_views={self.num_views}, err_max={self.err_max}, keep_orig={self.keep_orig} )'
        return repr_str
