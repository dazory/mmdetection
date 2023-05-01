import numpy as np
import torch
import kornia.filters as F
import kornia.geometry.transform as T

from ..builder import TRANSFORMATIONS
from .spatial_ops import (Rotate, Shear, Translate)


def sample_level(n):
    return np.random.uniform(0.1, n)


@TRANSFORMATIONS.register_module()
class BboxRotate(Rotate):
    def __init__(self, blur=None, *args, **kwargs):
        super(BboxRotate, self).__init__(*args, **kwargs)
        if blur is not None:
            assert isinstance(blur, dict), \
                f"blur should be a dict, but got {type(blur)} with value {blur}"
            self.blur = True
            self.Kx, self.Ky = blur['Kx'], blur['Ky']
            self.r = blur['r']
        else:
            self.blur = False

    def __call__(self, img, bboxes=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"
        assert bboxes is not None, "bboxes should be provided"

        for i, bbox in enumerate(bboxes):
            num_objs = bbox.shape[0]
            repeated_img = img[i].unsqueeze(0).repeat(num_objs, 1, 1, 1)

            masks = torch.zeros(
                (num_objs, 1, img.shape[-2], img.shape[-1]),
                device=img.get_device(), dtype=img.dtype)
            gaussian_kernels = torch.zeros(
                (num_objs, self.Kx, self.Ky), device=img.get_device(), dtype=img.dtype) \
                if self.blur else None
            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                masks[j, :, y1:y2, x1:x2] = 1
                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernels[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]
            if self.blur:
                masks = F.filter2d(masks, gaussian_kernels)

            angles = self._get_angles(num_objs, device=img.get_device(), dtype=img.dtype)
            centers = torch.stack(((bbox[:, 0] + bbox[:, 2]) / 2,
                                  (bbox[:, 1] + bbox[:, 3]) / 2), dim=1)
            rotated_img = T.rotate(repeated_img, angles, center=centers,
                                   mode=self.interpolation)

            for j in range(num_objs):
                img[i] = masks[j] * rotated_img[j] + (1-masks[j]) * img[i]

        return img


@TRANSFORMATIONS.register_module()
class BboxShear(Shear):
    def __init__(self, blur=None, *args, **kwargs):
        super(BboxShear, self).__init__(*args, **kwargs)
        if blur is not None:
            assert isinstance(blur, dict), \
                f"blur should be a dict, but got {type(blur)} with value {blur}"
            self.blur = True
            self.Kx, self.Ky = blur['Kx'], blur['Ky']
            self.r = blur['r']
        else:
            self.blur = False

    def __call__(self, img, bboxes=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"
        assert bboxes is not None, "bboxes should be provided"

        for i, bbox in enumerate(bboxes):
            num_objs = bbox.shape[0]
            repeated_img = img[i].unsqueeze(0).repeat(num_objs, 1, 1, 1)

            masks = torch.zeros(
                (num_objs, 1, img.shape[-2], img.shape[-1]),
                device=img.get_device(), dtype=img.dtype)
            gaussian_kernels = torch.zeros(
                (num_objs, self.Kx, self.Ky), device=img.get_device(), dtype=img.dtype) \
                if self.blur else None
            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                masks[j, :, y1:y2, x1:x2] = 1
                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernels[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]
            if self.blur:
                masks = F.filter2d(masks, gaussian_kernels)

            x_degrees = self._get_shearing_degrees(num_objs, device=img.get_device())
            y_degrees = self._get_shearing_degrees(num_objs, device=img.get_device())

            centers = torch.stack(((bbox[:, 0] + bbox[:, 2]) / 2,
                                   (bbox[:, 1] + bbox[:, 3]) / 2), dim=1)

            sheared_img = T.warp_affine(
                repeated_img,
                T.get_shear_matrix2d(centers, x_degrees, y_degrees)[:, :2, :],
                dsize=img.shape[-2:],
                mode=self.interpolation)

            for j in range(num_objs):
                img[i] = masks[j] * sheared_img[j] + (1-masks[j]) * img[i]

        return img


@TRANSFORMATIONS.register_module()
class BboxShearX(BboxShear):
    def __init__(self, *args, **kwargs):
        super(BboxShearX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class BboxShearY(BboxShear):
    def __init__(self, *args, **kwargs):
        super(BboxShearY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class BboxTranslate(Translate):
    def __init__(self, blur=None, *args, **kwargs):
        super(BboxTranslate, self).__init__(*args, **kwargs)
        if blur is not None:
            assert isinstance(blur, dict), \
                f"blur should be a dict, but got {type(blur)} with value {blur}"
            self.blur = True
            self.Kx, self.Ky = blur['Kx'], blur['Ky']
            self.r = blur['r']
        else:
            self.blur = False

    def __call__(self, img, bboxes=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"
        assert bboxes is not None, "bboxes should be provided"

        for i, bbox in enumerate(bboxes):
            num_objs = bbox.shape[0]
            repeated_img = img[i].unsqueeze(0).repeat(num_objs, 1, 1, 1)

            masks = torch.zeros(
                (num_objs, 1, img.shape[-2], img.shape[-1]),
                device=img.get_device(), dtype=img.dtype)
            gaussian_kernels = torch.zeros(
                (num_objs, self.Kx, self.Ky), device=img.get_device(), dtype=img.dtype) \
                if self.blur else None

            translation_x, translation_y = [], []
            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                masks[j, :, y1:y2, x1:x2] = 1

                translation_x.append(self._get_translation(x2 - x1))
                translation_y.append(self._get_translation(y2 - y1))

                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernels[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]

            if self.blur:
                masks = F.filter2d(masks, gaussian_kernels)

            translation_x = torch.tensor(translation_x, device=img.get_device())
            translation_y = torch.tensor(translation_y, device=img.get_device())
            translation = torch.stack([translation_x, translation_y], dim=-1).type(torch.float32)

            translated_img = T.translate(
                repeated_img,
                translation=translation,
                mode=self.interpolation)

            for j in range(num_objs):
                img[i] = masks[j] * translated_img[j] + (1-masks[j]) * img[i]

        return img


@TRANSFORMATIONS.register_module()
class BboxTranslateX(BboxTranslate):
    def __init__(self, *args, **kwargs):
        super(BboxTranslateX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class BboxTranslateY(BboxTranslate):
    def __init__(self, *args, **kwargs):
        super(BboxTranslateY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)
