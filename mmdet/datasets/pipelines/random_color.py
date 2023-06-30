# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
import math
import numpy as np
import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomColor:
    def __init__(self,
                 num_views=1, keep_orig=True,
                 cut_max=200, T_min=0.0, T_max=0.05,
                 use_mix=False, mixture_width=3, mixture_depth=-1, mixture_coeff=None,
                 use_oa=False, oa_version=None, use_blur=True, spatial_ratio=4, sigma_ratio=0.3,
                 ):
        super().__init__()
        self.num_views = num_views
        self.cut_max = cut_max
        self.T_min, self.T_max = T_min, T_max
        self.keep_orig = keep_orig
        if self.keep_orig and self.num_views < 2: warnings.warn(
            f'keep_orig==True but num_views is less than two. This has no effect.')

        self.use_mix = use_mix
        if use_mix:
            self.aug_prob_coeff = 1.0
            self.mixture_coeff = (1.0, 1.0) if mixture_coeff is None else mixture_coeff
            self.mixture_width = mixture_width
            self.mixture_depth = mixture_depth
        self.use_oa = use_oa
        if use_oa:
            self.mixture_coeff = (20.0, 5.0) if mixture_coeff is None else mixture_coeff
            self.use_blur = use_blur
            if use_blur:
                self.spatial_ratio = spatial_ratio  # Boost blurred mask generation
                self.sigma_ratio = sigma_ratio
        self.oa_version = oa_version

    def _generate_mask_gt_bboxes_list(self, img_shape, gt_bboxes):
        (h_img, w_img, c_img) = img_shape
        if self.use_blur:
            target_shape = tuple([h_img // self.spatial_ratio, w_img // self.spatial_ratio, c_img])
            sigma_list = [((gt_bbox[2] - gt_bbox[0]) // self.spatial_ratio * self.sigma_ratio / 3 * 2,
                           (gt_bbox[3] - gt_bbox[1]) // self.spatial_ratio * self.sigma_ratio / 3 * 2) for gt_bbox in gt_bboxes]
        else:
            target_shape = tuple([h_img, w_img, c_img])
            sigma_list = None

        self.mask_gt_bboxes_list = []
        for i, gt_bbox in enumerate(gt_bboxes):
            x1, y1, x2, y2 = np.array(gt_bbox // self.spatial_ratio, dtype=np.int32)
            mask_gt_bboxes = np.zeros(target_shape, dtype=np.float32)
            mask_gt_bboxes[y1:y2, x1:x2, :] = 1.0
            if self.use_blur:
                if (sigma_list[i][0] <= 0 or sigma_list[i][1] <= 0):
                    mask_gt_bboxes = None
                else:
                    mask_gt_bboxes = cv2.GaussianBlur(mask_gt_bboxes, (0, 0),
                                                      sigmaX=sigma_list[i][0], sigmaY=sigma_list[i][1])
            self.mask_gt_bboxes_list.append(cv2.resize(mask_gt_bboxes, (w_img, h_img)))

    def aug(self, img, gt_bboxes=None):
        assert np.max(img) <= 1 and 0 <=np.min(img), "img must be in [0, 1]"

        # Sample parameters
        cut = np.random.randint(2, self.cut_max)
        k_min = np.random.randint(1, cut)

        # Mask
        if self.use_oa:
            masks = np.zeros((len(gt_bboxes), *img.shape))
            for i, bbox in enumerate(gt_bboxes):
                mask = np.zeros(img.shape, dtype=np.uint8)
                x1, y1, x2, y2 = bbox
                sigma_x, sigma_y = (x2 - x1) / 6.0, (y2 - y1) / 6.0
                kernel_x, kernel_y = 2 * int(4 * sigma_x) + 1, 2 * int(4 * sigma_y) + 1
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 255, 255), thickness=-1)
                masks[i] = cv2.GaussianBlur(mask, (kernel_x, kernel_y), sigmaX=sigma_x, sigmaY=sigma_y)
            mask = np.max(masks, axis=0)

        # Augmentation
        freqs = np.zeros(img.shape, dtype=np.float32)
        for i in range(k_min, min(k_min+20, cut)):
            beta = np.random.rand(3) * np.sqrt(np.random.uniform(self.T_min, self.T_max))
            if self.use_oa:
                _freqs = beta * np.sin(math.pi * i * copy.deepcopy(img))
                if self.oa_version == 'none':
                    freqs += _freqs * (1.0 - mask / 255)
                elif self.oa_version == 'random':
                    if np.random.rand() < 0.5:
                        _freqs = _freqs * (1.0 - mask / 255)
                    freqs += _freqs
                elif self.oa_version == 'mix':
                    freqs += _freqs
                else:
                    raise NotImplementedError
            else:
                freqs += beta * np.sin(math.pi * i * copy.deepcopy(img))
        img_aug = np.clip(img + freqs, 0, 1)

        if self.use_mix:
            if self.use_oa:
                m = np.float32(np.random.beta(self.mixture_coeff[0], self.mixture_coeff[1]))
                mask_all = np.max(self.mask_gt_bboxes_list, axis=0)
                _img = m * img * mask_all + (1.0 - m) * img * (1.0 - mask_all)
                _img_aug = (1.0 - m) * img_aug * mask_all + m * img_aug * (1.0 - mask_all)
                img_augmix = _img + _img_aug
            else:
                m = np.float32(np.random.beta(self.mixture_coeff[0], self.mixture_coeff[1]))
                img_augmix = (1 - m) * img + m * img_aug
            return np.asarray(img_augmix, dtype=np.uint8)
        else:
            return np.asarray(img_aug, dtype=np.uint8)

    def __call__(self, results):
        img = results['img']
        ori_type = img.dtype
        if img.dtype == np.uint8:
            img = img / 255

        if self.use_oa:
            self._generate_mask_gt_bboxes_list(img_shape=results['img'].shape, gt_bboxes=results['gt_bboxes'])

        results['custom_fields'] = []

        i_min = 2 if self.keep_orig else 1
        for i in range(i_min, self.num_views + 1):
            img_aug = self.aug(copy.deepcopy(img), results['gt_bboxes'])
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
