import numpy as np
import torch
import kornia.filters as F
import kornia.geometry.transform as T

from ..builder import TRANSFORMATIONS
from .bbox_spatial_ops import (BboxRotate, BboxShear, BboxTranslate)


def sample_level(n):
    return np.random.uniform(0.1, n)


@TRANSFORMATIONS.register_module()
class NotBboxRotate(BboxRotate):
    def __init__(self, *args, **kwargs):
        super(NotBboxRotate, self).__init__(*args, **kwargs)

    def __call__(self, img, bboxes=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"
        assert bboxes is not None, "bboxes should be provided"

        batch_size = img.shape[0]
        masks = torch.zeros((batch_size, 1, img.shape[-2], img.shape[-1]),
                            device=img.get_device(), dtype=img.dtype)
        for i, bbox in enumerate(bboxes):
            num_objs = bbox.shape[0]

            mask = torch.zeros((num_objs, 1, img.shape[-2], img.shape[-1]),
                               device=img.get_device(), dtype=img.dtype)
            gaussian_kernel = torch.zeros((num_objs, self.Kx, self.Ky),
                                          device=img.get_device(), dtype=img.dtype) \
                if self.blur else None
            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                mask[j, :, y1:y2, x1:x2] = 1
                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernel[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]
            if self.blur:
                mask = F.filter2d(mask, gaussian_kernel)

            masks[i] = torch.max(mask, dim=0).values

        # Rotate the mask and image
        angles = self._get_angles(batch_size, device=img.get_device(), dtype=img.dtype)
        rotated_masks = T.rotate(masks, angles, center=None, mode=self.interpolation)
        rotated_imgs = T.rotate(img, angles, center=None, mode=self.interpolation)

        # Composite the original image and the rotated image
        sum_masks = torch.stack((masks, rotated_masks), dim=0)
        sum_masks = torch.max(sum_masks, dim=0).values
        img = sum_masks * img + (1-sum_masks) * rotated_imgs

        return img


@TRANSFORMATIONS.register_module()
class NotBboxShear(BboxShear):
    def __init__(self, *args, **kwargs):
        super(NotBboxShear, self).__init__(*args, **kwargs)

    def __call__(self, img, bboxes=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"
        assert bboxes is not None, "bboxes should be provided"

        batch_size = img.shape[0]
        masks = torch.zeros((batch_size, 1, img.shape[-2], img.shape[-1]),
                            device=img.get_device(), dtype=img.dtype)
        for i, bbox in enumerate(bboxes):
            num_objs = bbox.shape[0]

            mask = torch.zeros(
                (num_objs, 1, img.shape[-2], img.shape[-1]),
                device=img.get_device(), dtype=img.dtype)
            gaussian_kernel = torch.zeros(
                (num_objs, self.Kx, self.Ky), device=img.get_device(), dtype=img.dtype) \
                if self.blur else None
            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                mask[j, :, y1:y2, x1:x2] = 1
                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernel[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]
            if self.blur:
                mask = F.filter2d(mask, gaussian_kernel)

            masks[i] = torch.max(mask, dim=0).values

        # Shear the mask and image
        x_degrees = self._get_shearing_degrees(batch_size, device=img.get_device())
        y_degrees = self._get_shearing_degrees(batch_size, device=img.get_device())
        centers = torch.tensor(((img.shape[-1] - 1) / 2., (img.shape[-2] - 1) / 2.),
                               device=img.get_device()).repeat(batch_size, 1)
        shear_matrices = T.get_shear_matrix2d(centers, x_degrees, y_degrees)[:, :2, :]
        sheared_masks = T.warp_affine(masks, shear_matrices, dsize=img.shape[-2:], mode=self.interpolation)
        sheared_imgs = T.warp_affine(img, shear_matrices, dsize=img.shape[-2:], mode=self.interpolation)

        # Composite the original image and the rotated image
        sum_masks = torch.stack((masks, sheared_masks), dim=0)
        sum_masks = torch.max(sum_masks, dim=0).values
        img = sum_masks * img + (1 - sum_masks) * sheared_imgs

        return img


@TRANSFORMATIONS.register_module()
class NotBboxShearX(NotBboxShear):
    def __init__(self, *args, **kwargs):
        super(NotBboxShearX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class NotBboxShearY(NotBboxShear):
    def __init__(self, *args, **kwargs):
        super(NotBboxShearY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class NotBboxTranslate(BboxTranslate):
    def __init__(self, *args, **kwargs):
        super(NotBboxTranslate, self).__init__(*args, **kwargs)

    def __call__(self, img, bboxes=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"
        assert bboxes is not None, "bboxes should be provided"

        batch_size = img.shape[0]
        masks = torch.zeros((batch_size, 1, img.shape[-2], img.shape[-1]),
                            device=img.get_device(), dtype=img.dtype)
        for i, bbox in enumerate(bboxes):
            num_objs = bbox.shape[0]

            mask = torch.zeros((num_objs, 1, img.shape[-2], img.shape[-1]),
                               device=img.get_device(), dtype=img.dtype)
            gaussian_kernel = torch.zeros(
                (num_objs, self.Kx, self.Ky), device=img.get_device(), dtype=img.dtype) \
                if self.blur else None

            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                mask[j, :, y1:y2, x1:x2] = 1

                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernel[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]

            if self.blur:
                mask = F.filter2d(mask, gaussian_kernel)

            masks[i] = torch.max(mask, dim=0).values

        # Translate the mask and image
        translations_x = self._get_translations(batch_size, img.shape[-1], device=img.get_device())
        translations_y = self._get_translations(batch_size, img.shape[-2], device=img.get_device())
        translations = torch.stack([translations_x, translations_y], dim=-1).type(torch.float32)
        translated_masks = T.translate(masks, translation=translations, mode=self.interpolation)
        translated_imgs = T.translate(img, translation=translations, mode=self.interpolation)

        # Composite the original image and the translated image
        sum_masks = torch.stack((masks, translated_masks), dim=0)
        sum_masks = torch.max(sum_masks, dim=0).values
        img = sum_masks * img + (1 - sum_masks) * translated_imgs

        return img


@TRANSFORMATIONS.register_module()
class NotBboxTranslateX(NotBboxTranslate):
    def __init__(self, *args, **kwargs):
        super(NotBboxTranslateX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class NotBboxTranslateY(NotBboxTranslate):
    def __init__(self, *args, **kwargs):
        super(NotBboxTranslateY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)
