# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import cv2
import numpy as np
from numpy import random

from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from PIL import Image, ImageOps, ImageEnhance
from .augmix import (int_parameter, sample_level, float_parameter)


def autocontrast(pil_img, **kwargs):
  return ImageOps.autocontrast(pil_img)

def equalize(pil_img, **kwargs):
  return ImageOps.equalize(pil_img)

def posterize(pil_img, level, use_random=True, **kwargs):
    level = int_parameter(sample_level(level) if use_random else level, maxval=3)
    return ImageOps.posterize(pil_img, 4 - level)

def solarize(pil_img, level, use_random=True, **kwargs):
  level = int_parameter(sample_level(level) if use_random else level, maxval=256)
  return ImageOps.solarize(pil_img, 256 - level)

def color(pil_img, level, use_random=True, **kwargs):
    level = float_parameter(sample_level(level) if use_random else level, maxval=.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)

def contrast(pil_img, level, use_random=True, **kwargs):
    level = float_parameter(sample_level(level) if use_random else level, 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

def brightness(pil_img, level, use_random=True, **kwargs):
    level = float_parameter(sample_level(level) if use_random else level, 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)

def sharpness(pil_img, level, use_random=True, **kwargs):
    level = float_parameter(sample_level(level) if use_random else level, 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


@PIPELINES.register_module()
class Color:
    def __init__(self, version='0',
                 num_views=2, keep_orig=True,
                 use_mix=False,
                 use_oa=False, use_blur=True, spatial_ratio=4, sigma_ratio=0.3,
                 severity=3,
                 ):
        self.version = version
        self.aug_list = self._get_augs()

        self.num_views = num_views
        self.keep_orig = keep_orig
        if self.num_views == 1 and self.keep_orig:
            warnings.warn('No augmentation will be applied since num_views=1 and keep_orig=True')

        self.use_mix = use_mix
        if use_mix:
            self.aug_prob_coeff = 1.0
            self.mixture_width = 3
            self.mixture_depth = -1
        self.use_oa = use_oa
        if use_oa:
            self.use_blur = use_blur
            if use_blur:
                self.spatial_ratio = spatial_ratio  # Boost blurred mask generation
                self.sigma_ratio = sigma_ratio

        self.severity = severity

    def _get_augs(self):
        if self.version == '0':
            return [autocontrast, equalize, posterize, solarize]
        elif self.version == 'color_all':
            return [autocontrast, equalize, posterize, solarize,
                    color, contrast, brightness, sharpness]
        else:
            raise NotImplementedError

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

    def __call__(self, results):
        if self.use_oa:
            self._generate_mask_gt_bboxes_list(img_shape=results['img'].shape, gt_bboxes=results['gt_bboxes'])

        results['custom_fields'] = []
        for i in range(1, self.num_views + 1):
            if i == 1:
                if not self.keep_orig:
                    img_aug = self.aug(results['img'].copy())
                    results['dx'] = np.mean(np.abs(results['img'] - img_aug))
                    results['img'] = img_aug
                    results['custom_fields'] += ['dx']
                results['img_fields'] = ['img']
            else:
                results[f'img{i}'] = self.aug(results['img'].copy())
                results['img_fields'] += [f'img{i}']
                results[f'dx{i}'] = np.mean(np.abs(results['img'] - results[f'img{i}']))
                results['custom_fields'] += [f'img{i}', f'dx{i}']
        return results

    def aug(self, img):
        h_img, w_img, _ = img.shape
        img_size = (w_img, h_img)
        if self.use_mix:
            # Sample parameters
            ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
            m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

            img_mix = np.zeros_like(img.copy(), dtype=np.float32)
            for i in range(self.mixture_width):
                depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
                img_aug = Image.fromarray(img.copy(), "RGB")
                for _ in range(depth):
                    img_aug = self._aug(img_aug, img_size)
                img_mix += ws[i] * np.asarray(img_aug, dtype=np.float32)

            img_augmix = (1 - m) * img + m * img_mix
            return img_augmix.astype(np.uint8)
        else:
            img_aug = self._aug(Image.fromarray(img.copy(), "RGB"), img_size)
            if self.use_oa:
                e = 0
                severity = 5
                pil_img = self.aug_list[2](Image.fromarray(img[...,::-1].copy(), "RGB"),
                                           level=severity, img_size=img_size, use_random=False)
                Image.fromarray(img[...,::-1].copy(), "RGB").save(f'/ws/data/dshong/mmdetection/visualization/scale_effect_run2/mix_bbox/{e}.0.orig.png')
                pil_img.save(f'/ws/data/dshong/mmdetection/visualization/scale_effect_run2/mix_bbox/{e}.0.aug.png')
                for m in [0.0, 0.2, 0.5, 0.8, 1.0]:
                    mask_all = np.max(self.mask_gt_bboxes_list, axis=0)
                    np_img = np.asarray(pil_img, dtype=np.uint8)
                    _img0 = np_img * ((1.0 - m) * (1.0 - mask_all))
                    _img1 = np_img * (1.0 - m)
                    _img2 = np_img * ((1.0 - m) * (1.0 - mask_all))

                    np_img = img * (m * mask_all) +  np_img * ((1.0 - m) * (1.0 - mask_all))
                    Image.fromarray(np.asarray(np_img[..., ::-1], dtype=np.uint8)).\
                        save(f'/ws/data/dshong/mmdetection/visualization/scale_effect_run2/mix_bbox/{e}.1.mix_bbox_m{m:.1f}.png')
            return img_aug.astype(np.uint8)

    def _aug(self, img, img_size):
        op = np.random.choice(self.aug_list)
        pil_img = op(img, level=self.severity, img_size=img_size)
        return np.asarray(pil_img, dtype=np.uint8)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
