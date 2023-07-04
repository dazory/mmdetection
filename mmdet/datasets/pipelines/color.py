# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import scipy
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
                 use_mix=False, mixture_width=3, mixture_depth=-1, mixture_coeff=None,
                 use_oa=False, oa_version='none', use_blur=True, spatial_ratio=4, sigma_ratio=0.3,
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
            self.mixture_coeff = (1.0, 1.0) if mixture_coeff is None else mixture_coeff
            self.mixture_width = mixture_width
            self.mixture_depth = mixture_depth
        self.use_oa = use_oa
        if use_oa:
            self.oa_version=oa_version
            if oa_version == 'scale':
                self.scale_thresholds = [(0, 7), (7, 14), (14, 28), (28, 56), (56, 112), (112, 224), (224, 448), (448, 100000)]
                self.m_mins = [1.0 - i / len(self.scale_thresholds) for i in range(1, len(self.scale_thresholds)+1)]
            elif oa_version == 'meanstd':
                self.scale_thresholds = [(0, 2), (2, 4), (4, 6), (6, 100000)]
                self.m_mins = [1.0 - i / len(self.scale_thresholds) for i in range(1, len(self.scale_thresholds) + 1)]
            elif oa_version == 'std':
                self.scale_thresholds = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 100000)]
                self.m_mins = [1.0 - i / len(self.scale_thresholds) for i in range(1, len(self.scale_thresholds) + 1)]
            elif oa_version == 'saliency':
                self.scale_thresholds = [(0, 5), (5, 10), (10, 20), (20, 100000)]
                self.m_mins = [1.0 - i / len(self.scale_thresholds) for i in range(1, len(self.scale_thresholds) + 1)]
            self.mixture_coeff = (20.0, 5.0) if mixture_coeff is None else mixture_coeff
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
                if not (sigma_list[i][0] <= 0 or sigma_list[i][1] <= 0):
                    mask_gt_bboxes = cv2.GaussianBlur(mask_gt_bboxes, (0, 0),
                                                      sigmaX=sigma_list[i][0], sigmaY=sigma_list[i][1])
                self.mask_gt_bboxes_list.append(cv2.resize(mask_gt_bboxes, (w_img, h_img)))

        if self.oa_version == 'scale':
            scales = np.sqrt(
                (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            )
            self.scale_gt_bboxes_list = list(scales)

    def __call__(self, results):
        if self.use_oa:
            self._generate_mask_gt_bboxes_list(img_shape=results['img'].shape, gt_bboxes=results['gt_bboxes'])
            if self.mixture_coeff == -1:
                self.gt_bboxes = results['gt_bboxes']

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
            if isinstance(self.mixture_coeff, tuple):
                # Sample parameters
                ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
                m = np.float32(np.random.beta(self.mixture_coeff[0], self.mixture_coeff[1]))

                img_mix = np.zeros_like(img.copy(), dtype=np.float32)
                for i in range(self.mixture_width):
                    depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
                    img_aug = Image.fromarray(img.copy(), "RGB")
                    for _ in range(depth):
                        img_aug = self._aug(img_aug, img_size)
                    img_mix += ws[i] * np.asarray(img_aug, dtype=np.float32)

                if self.use_oa:
                    mask_all = np.max(self.mask_gt_bboxes_list, axis=0)
                    img = m * img * mask_all + (1.0 - m) * img * (1.0 - mask_all)
                    img_mix = (1.0 - m) * img_mix * mask_all + m * img_mix * (1.0 - mask_all)
                    img_augmix = img + img_mix
                else:
                    img_augmix = (1 - m) * img + m * img_mix
                return np.asarray(img_augmix, dtype=np.uint8)
            elif self.use_oa and self.mixture_coeff == -1:
                # Sample parameters
                ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
                img_mix = np.zeros_like(img.copy(), dtype=np.float32)
                for i in range(self.mixture_width):
                    depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
                    img_aug = Image.fromarray(img.copy(), "RGB")
                    for _ in range(depth):
                        img_aug = self._aug(img_aug, img_size)
                    img_mix += ws[i] * np.asarray(img_aug, dtype=np.float32)

                if self.oa_version == 'dist':
                    img_gray = cv2.cvtColor(img_mix, cv2.COLOR_RGB2GRAY)
                    img_gray_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    for (gt_bbox, mask_gt_bbox) in zip(self.gt_bboxes, self.mask_gt_bboxes_list):
                        x1, y1, x2, y2 = np.asarray(gt_bbox, dtype=np.uint32)
                        bbox_img = img_gray[y1:y2, x1:x2].flatten()
                        hist, _ = np.histogram(bbox_img, bins=128)
                        prob_dist = hist / hist.sum()
                        entropy = scipy.stats.entropy(prob_dist, base=2)
                        raise NotImplementedError()
                elif self.oa_version == 'scale':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    m_list = []
                    for scale in self.scale_gt_bboxes_list:
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= scale and scale < scale_thrs[1]:
                                m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_mix, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_mix * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list, axis=0) # np.clip(np.sum(self.mask_gt_bboxes_list, axis=0), 0, 1)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_mix * (1.0 - mask_all)

                    img, img_mix = orig, aug
                elif self.oa_version == 'meanstd':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    img_gray_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    m_list = []
                    for gt_bbox in self.gt_bboxes:
                        x1, y1, x2, y2 = np.asarray(gt_bbox, dtype=np.uint32)
                        bbox_img = img_gray_orig[y1:y2, x1:x2].flatten()
                        mean, std = np.mean(bbox_img), np.std(bbox_img)
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= mean/std and mean/std < scale_thrs[1]:
                                m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_mix, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_mix * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list, axis=0)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_mix * (1.0 - mask_all)

                    img, img_mix = orig, aug
                elif self.oa_version == 'std':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    img_gray_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    m_list = []
                    for gt_bbox in self.gt_bboxes:
                        x1, y1, x2, y2 = np.asarray(gt_bbox, dtype=np.uint32)
                        bbox_img = img_gray_orig[y1:y2, x1:x2].flatten()
                        std = np.std(bbox_img)
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= std and std < scale_thrs[1]:
                                m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_mix, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_mix * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list, axis=0)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_mix * (1.0 - mask_all)

                    img, img_mix = orig, aug
                elif self.oa_version == 'saliency':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    m_list = []
                    for gt_bbox in self.gt_bboxes:
                        x1, y1, x2, y2 = np.asarray(gt_bbox, dtype=np.uint32)
                        bbox_img = img[y1:y2, x1:x2]
                        saliency = cv2.saliency.StaticSaliencySpectralResidual_create() # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                        (success, saliency_map) = saliency.computeSaliency(bbox_img)
                        saliency_score = np.mean((saliency_map * 255).astype("uint8"))
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= saliency_score and saliency_score < scale_thrs[1]:
                                m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_mix, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_mix * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list, axis=0)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_mix * (1.0 - mask_all)

                    img, img_mix = orig, aug

                img_augmix = img + img_mix

                return np.asarray(img_augmix, dtype=np.uint8)
            else:
                raise NotImplementedError
        else:
            img_aug = self._aug(Image.fromarray(img.copy(), "RGB"), img_size)
            return np.asarray(img_aug, dtype=np.uint8)

    def _aug(self, img, img_size):
        op = np.random.choice(self.aug_list)
        pil_img = op(img, level=self.severity, img_size=img_size)
        return pil_img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
