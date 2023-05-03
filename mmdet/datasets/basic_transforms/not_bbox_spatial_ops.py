import numpy as np
import torch
import kornia.filters as F
import kornia.geometry.transform as T

from ..builder import TRANSFORMATIONS
from .spatial_ops import (Rotate, Shear, Translate)


def sample_level(n):
    return np.random.uniform(0.1, n)


@TRANSFORMATIONS.register_module()
class NotBboxRotate(Rotate):
    def __init__(self, blur=None, *args, **kwargs):
        super(NotBboxRotate, self).__init__(*args, **kwargs)
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
                    # sigma_x, sigma_y = min((x2 - x1) * self.r / 3.0, (self.Kx - 1) / 6.0), \
                    #                    min((y2 - y1) * self.r / 3.0, (self.Ky - 1) / 6.0)
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    print(f"sigma (x, y) = ({sigma_x:.3f}, {sigma_y:.3f})")
                    gaussian_kernels[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]
            if self.blur:
                masks = F.filter2d(masks, gaussian_kernels)

            masks = torch.max(masks, dim=0).values

            # Rotate the mask and image
            angle = torch.tensor(self._get_angle(), device=img.get_device(), dtype=img.dtype)
            rotated_masks = T.rotate(masks, angle, center=None, mode=self.interpolation)
            rotated_img = T.rotate(img[i].unsqueeze(0), angle, center=None, mode=self.interpolation)

            # Composite the original image and the rotated image
            sum_masks = torch.stack((masks, rotated_masks), dim=0)
            sum_masks = torch.max(sum_masks, dim=0).values
            img[i] = sum_masks * img[i] + (1-sum_masks) * rotated_img[0]

        return img


@TRANSFORMATIONS.register_module()
class NotBboxShear(Shear):
    def __init__(self, blur=None, *args, **kwargs):
        super(NotBboxShear, self).__init__(*args, **kwargs)
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

            mask = torch.zeros(
                (num_objs, 1, img.shape[-2], img.shape[-1]),
                device=img.get_device(), dtype=img.dtype)
            gaussian_kernels = torch.zeros(
                (num_objs, self.Kx, self.Ky), device=img.get_device(), dtype=img.dtype) \
                if self.blur else None
            for j in range(num_objs):
                x1, y1, x2, y2 = bbox[j].type(torch.int32)
                mask[j, :, y1:y2, x1:x2] = 1
                if self.blur:
                    sigma_x, sigma_y = (x2 - x1) * self.r / 3.0, (y2 - y1) * self.r / 3.0
                    gaussian_kernels[j] = F.get_gaussian_kernel2d((self.Kx, self.Ky), (sigma_x, sigma_y))[0]
            if self.blur:
                mask = F.filter2d(mask, gaussian_kernels)

            mask = torch.max(mask, dim=0).values

            # Shear the mask and image
            x_degree = torch.tensor(self._get_shearing_degree(), device=img.get_device()).unsqueeze(0)
            y_degree = torch.tensor(self._get_shearing_degree(), device=img.get_device()).unsqueeze(0)
            center = torch.tensor(((img.shape[-1] - 1) / 2., (img.shape[-2] - 1) / 2.), device=img.get_device()).unsqueeze(0)
            shear_matrix = T.get_shear_matrix2d(center, x_degree, y_degree)[:, :2, :]
            sheared_mask = T.warp_affine(mask.unsqueeze(0), shear_matrix, dsize=img.shape[-2:], mode=self.interpolation)
            sheared_img = T.warp_affine(img[i].unsqueeze(0), shear_matrix, dsize=img.shape[-2:], mode=self.interpolation)

            # Composite the original image and the rotated image
            sum_mask = torch.stack((mask, sheared_mask[0]), dim=0)
            sum_mask = torch.max(sum_mask, dim=0).values
            img[i] = sum_mask * img[i] + (1 - sum_mask) * sheared_img[0]

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
class NotBboxTranslate(Translate):
    def __init__(self, blur=None, *args, **kwargs):
        super(NotBboxTranslate, self).__init__(*args, **kwargs)
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

            masks = torch.zeros((num_objs, 1, img.shape[-2], img.shape[-1]),
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

            masks = torch.max(masks, dim=0).values

            # Translate the mask and image
            translation_x = torch.tensor(self._get_translation(img.shape[-1]), device=img.get_device())
            translation_y = torch.tensor(self._get_translation(img.shape[-2]), device=img.get_device())
            translation = torch.stack([translation_x, translation_y], dim=-1).type(torch.float32).unsqueeze(0)
            translated_masks = T.translate(masks.unsqueeze(0), translation=translation, mode=self.interpolation)
            translated_img = T.translate(img[i].unsqueeze(0), translation=translation, mode=self.interpolation)

            # Composite the original image and the translated image
            sum_masks = torch.stack((masks, translated_masks[0]), dim=0)
            sum_masks = torch.max(sum_masks, dim=0).values
            img[i] = sum_masks * img[i] + (1 - sum_masks) * translated_img[0]

        return img


@TRANSFORMATIONS.register_module()
class NotBboxTranslateX(NotBboxTranslate):
    def __init__(self, *args, **kwargs):
        super(NotBboxTranslateX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class NotBboxTranslateY(NotBboxTranslate):
    def __init__(self, *args, **kwargs):
        super(NotBboxTranslateY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)
