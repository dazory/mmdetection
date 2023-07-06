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
            self.oa_version = oa_version
            if oa_version == 'scale':
                self.scale_thresholds = [(0, 7), (7, 14), (14, 28), (28, 56), (56, 112), (112, 224), (224, 448), (448, 100000)]
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

    def aug(self, img, gt_bboxes=None):
        assert np.max(img) <= 1 and 0 <=np.min(img), "img must be in [0, 1]"

        # Sample parameters
        cut = np.random.randint(2, self.cut_max)
        k_min = np.random.randint(1, cut)

        # Augmentation
        freqs = np.zeros(img.shape, dtype=np.float32)
        for i in range(k_min, min(k_min+20, cut)):
            beta = np.random.rand(3) * np.sqrt(np.random.uniform(self.T_min, self.T_max))
            freqs += beta * np.sin(math.pi * i * copy.deepcopy(img))
        img_aug = np.clip(img + freqs, 0, 1)

        _img = img.copy()
        _img_mix = img_aug.copy()
        # img = _img.copy()
        # img_aug = _img_mix.copy()
        if self.use_mix:
            if self.use_oa:
                if self.oa_version == 'saliency':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    m_list = []
                    for gt_bbox in self.gt_bboxes:
                        x1, y1, x2, y2 = np.asarray(gt_bbox, dtype=np.uint32)
                        if x2 - x1 < 1 or y2 - y1 < 1:
                            m_list.append(0.0)
                            continue
                        bbox_img = img[y1:y2, x1:x2]
                        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()  # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                        (success, saliency_map) = saliency.computeSaliency(bbox_img)
                        saliency_score = np.mean((saliency_map * 255).astype("uint8"))
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= saliency_score and saliency_score < scale_thrs[1]:
                                m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_aug, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_aug * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list, axis=0) # np.max(self.mask_gt_bboxes_list, axis=0)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_aug * (1.0 - mask_all)

                    img, img_aug = orig, aug
                elif self.oa_version == 'scale':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    m_list = []
                    for scale in self.scale_gt_bboxes_list:
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= scale and scale < scale_thrs[1]:
                                if self.use_mrange:
                                    m_list.append(np.float32(np.random.uniform(m_min, (1.0 - m_min) / 2.0)))
                                else:
                                    m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_aug, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_aug * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list,
                                      axis=0)  # np.clip(np.sum(self.mask_gt_bboxes_list, axis=0), 0, 1)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_aug * (1.0 - mask_all)

                    img, img_aug = orig, aug
                elif self.oa_version == 'std':
                    img = np.asarray(img, dtype=np.float32)

                    # Sample m parameter according to scale
                    img_gray_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    m_list = []
                    for gt_bbox in self.gt_bboxes:
                        x1, y1, x2, y2 = np.asarray(gt_bbox, dtype=np.uint32)
                        if x2 - x1 < 1 or y2 - y1 < 1:
                            m_list.append(0.0)
                            continue
                        bbox_img = img_gray_orig[y1:y2, x1:x2].flatten()
                        std = np.std(bbox_img)
                        for scale_thrs, m_min in zip(self.scale_thresholds, self.m_mins):
                            if scale_thrs[0] <= std and std < scale_thrs[1]:
                                if self.use_mrange:
                                    m_list.append(np.float32(np.random.uniform(m_min, (1.0 - m_min) / 2.0)))
                                else:
                                    m_list.append(np.float32(np.random.uniform(m_min, 1.0)))
                                break

                    # Mix orig and aug for gt_bbox
                    orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img_aug, dtype=np.float32)
                    for mask_gt_bbox, m in zip(self.mask_gt_bboxes_list, m_list):
                        orig += m * img * mask_gt_bbox
                        aug += (1.0 - m) * img_aug * mask_gt_bbox

                    # Mix orig and aug for background
                    mask_all = np.sum(self.mask_gt_bboxes_list, axis=0)
                    m = np.random.beta(1.0, 1.0)
                    orig += m * img * (1.0 - mask_all)
                    aug += (1.0 - m) * img_aug * (1.0 - mask_all)

                    img, img_aug = orig, aug

                img_augmix = img + img_aug

            else:
                m = np.float32(np.random.beta(self.mixture_coeff[0], self.mixture_coeff[1]))
                img_augmix = (1 - m) * img + m * img_aug
            return img_augmix
        else:
            return img_aug

    def __call__(self, results):
        img = results['img']
        ori_type = img.dtype
        if img.dtype == np.uint8:
            img = img / 255

        if self.use_oa:
            self._generate_mask_gt_bboxes_list(img_shape=results['img'].shape, gt_bboxes=results['gt_bboxes'])
            self.gt_bboxes = results['gt_bboxes']

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
