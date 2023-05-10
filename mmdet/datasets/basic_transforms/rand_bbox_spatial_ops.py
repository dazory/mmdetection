import numpy as np
import torch

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import TRANSFORMATIONS
from .bbox_spatial_ops import (BboxRotate, BboxShear, BboxTranslate)


def sample_level(n):
    return np.random.uniform(0.1, n)


def _generate_random_bboxes(img_shape, bboxes,
                            rand_type='bg', max_num_bboxes=(3, 10),
                            scales=(0.01, 0.2), ratios=(0.3, 1 / 0.3),
                            max_iters=100, eps=1e-6):
    max_num_bboxes = np.random.randint(*max_num_bboxes) \
        if isinstance(max_num_bboxes, tuple) else max_num_bboxes

    device, dtype = bboxes.device, bboxes.dtype
    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu().detach().numpy()
    (img_height, img_width) = img_shape

    random_bboxes = np.zeros((max_num_bboxes, 4), dtype=np.int32)
    num_bboxes = 0
    for _ in range(max_iters):
        if num_bboxes >= max_num_bboxes:
            break
        # Randomly sample a bbox
        x1, y1 = np.random.randint(0, img_width), np.random.randint(0, img_height)
        scale = np.random.uniform(*scales) * img_height * img_width
        ratio = np.random.uniform(*ratios)
        bbox_w, bbox_h = int(np.sqrt(scale / ratio)), int(np.sqrt(scale * ratio))

        # Compute IoU between the sampled bbox and all bboxes
        random_bbox = np.array([[x1, y1, min(x1 + bbox_w, img_width), min(y1 + bbox_h, img_height)]])
        ious = bbox_overlaps(random_bbox, bboxes)#

        if rand_type == 'bg':
            # Reject the sampled bbox if it has an IoU with any gt bbox above the threshold
            if np.sum(ious) > eps:
                continue
        elif rand_type == 'fg':
            # Reject the sampled bbox if it has no IoU with any gt bbox above the threshold
            if np.sum(ious) < eps:
                continue
            diff_bboxes = random_bbox - bboxes
            diff_bboxes[:, :2] = (diff_bboxes[:, :2] > 0)
            diff_bboxes[:, 2:] = (diff_bboxes[:, 2:] < 0)
            diff_mask = diff_bboxes.sum(axis=1) < 4
            if diff_mask.all():
                continue
        else:
            raise NotImplementedError

        # Accept the sampled bbox
        random_bboxes[num_bboxes, :] = random_bbox[0]
        num_bboxes += 1

    return torch.tensor(random_bboxes[:num_bboxes, :], device=device, dtype=dtype)


@TRANSFORMATIONS.register_module()
class RandBboxRotate(BboxRotate):
    def __init__(self, rand_cfg, *args, **kwargs):
        super(RandBboxRotate, self).__init__(*args, **kwargs)
        self.rand_cfg = rand_cfg

    def __call__(self, img, bboxes=None, **kwargs):
        random_bboxes = []
        for bbox in bboxes:
            random_bboxes.append(_generate_random_bboxes(
                img.shape[-2:], bbox, **self.rand_cfg))
        return super(RandBboxRotate, self).__call__(img, random_bboxes, **kwargs)


@TRANSFORMATIONS.register_module()
class RandBboxShear(BboxShear):
    def __init__(self, rand_cfg, *args, **kwargs):
        super(RandBboxShear, self).__init__(*args, **kwargs)
        self.rand_cfg = rand_cfg

    def __call__(self, img, bboxes=None, **kwargs):
        random_bboxes = []
        for bbox in bboxes:
            random_bboxes.append(_generate_random_bboxes(
                img.shape[-2:], bbox, **self.rand_cfg))
        return super(RandBboxShear, self).__call__(img, random_bboxes, **kwargs)


@TRANSFORMATIONS.register_module()
class RandBboxShearX(RandBboxShear):
    def __init__(self, *args, **kwargs):
        super(RandBboxShearX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class RandBboxShearY(RandBboxShear):
    def __init__(self, *args, **kwargs):
        super(RandBboxShearY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class RandBboxTranslate(BboxTranslate):
    def __init__(self, rand_cfg, *args, **kwargs):
        super(RandBboxTranslate, self).__init__(*args, **kwargs)
        self.rand_cfg = rand_cfg

    def __call__(self, img, bboxes=None, **kwargs):
        random_bboxes = []
        for bbox in bboxes:
            random_bboxes.append(_generate_random_bboxes(
                img.shape[-2:], bbox, **self.rand_cfg))
        return super(RandBboxTranslate, self).__call__(img, random_bboxes, **kwargs)


@TRANSFORMATIONS.register_module()
class RandBboxTranslateX(RandBboxTranslate):
    def __init__(self, *args, **kwargs):
        super(RandBboxTranslateX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class RandBboxTranslateY(RandBboxTranslate):
    def __init__(self, *args, **kwargs):
        super(RandBboxTranslateY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)
