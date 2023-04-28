import numpy as np
import torch
import kornia.geometry.transform as T

from ..builder import TRANSFORMATIONS


def sample_level(n):
    return np.random.uniform(0.1, n)


@TRANSFORMATIONS.register_module()
class Rotate:
    def __init__(self, level,
                 randomness=True,
                 padding_mode='zeros',
                 interpolation='bilinear',):
        self.level = level
        self.randomness = randomness

        assert padding_mode in ['zeros', 'border', 'reflection'], \
            f"padding_mode should be 'zeros', 'border' or 'reflection', but got {padding_mode}"
        self.padding_mode = padding_mode

        assert interpolation in ['bilinear', 'nearest'], \
            f"interpolation should be 'bilinear' or 'nearest', but got {interpolation}"
        self.interpolation = interpolation

    def _get_angle(self):
        max_val = 30
        degrees = int(sample_level(self.level) * max_val / 10) \
            if self.randomness else int(self.level * max_val / 10)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return degrees

    def _get_angles(self, batch_size, device='cuda', dtype=torch.float32):
        return torch.tensor(
            [self._get_angle() for _ in range(batch_size)],
            device=device, dtype=dtype)

    def __call__(self, img, center=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"

        if center is not None:
            assert len(center.shape) == 2 and \
                center.shape[0] == img.shape[0] and \
                center.shape[1] == 2, \
                f"center shape should be (bs, 2), but got {center.shape}"

        bs = img.shape[0]
        angles = self._get_angles(bs, device=img.get_device(), dtype=img.dtype)

        rotated_img = T.rotate(img, angles, center=center,
                               mode=self.interpolation,
                               padding_mode=self.padding_mode)
        return rotated_img

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Shear:
    def __init__(self, level,
                 randomness=True,
                 x_axis=True, y_axis=False,
                 padding_mode='zeros',
                 interpolation='bilinear',):
        self.level = level
        self.randomness = randomness
        self.x_axis, self.y_axis = x_axis, y_axis

        assert padding_mode in ['zeros', 'border', 'reflection', 'fill'], \
            f"padding_mode should be 'zeros', 'border' or 'reflection', but got {padding_mode}"
        self.padding_mode = padding_mode

        assert interpolation in ['bilinear', 'nearest'], \
            f"interpolation should be 'bilinear' or 'nearest', but got {interpolation}"
        self.interpolation = interpolation

    def _get_shearing_degree(self):
        max_val = 0.3
        level = float(sample_level(self.level)) * max_val / 10. \
            if self.randomness else float(self.level) * max_val / 10.
        if np.random.uniform() > 0.5:
            level = -level
        return level

    def _get_shearing_degrees(self, batch_size,device='cuda'):
        return torch.tensor([
            self._get_shearing_degree() for _ in range(batch_size)],
            device=device)

    def __call__(self, img, center=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"

        bs = img.shape[0]
        device = img.get_device()

        x_degrees = self._get_shearing_degrees(bs, device) \
            if self.x_axis else torch.tensor([0.0], device=device).repeat(bs)
        y_degrees = self._get_shearing_degrees(bs, device) \
            if self.y_axis else torch.tensor([0.0], device=device).repeat(bs)
        if center is None:
            center = torch.tensor(img.shape[-2:], device=device) / 2

        shear_mat = T.get_shear_matrix2d(center, sx=x_degrees, sy=y_degrees)

        sheared_img = T.warp_affine(img, shear_mat[:, :2, :], dsize=img.shape[-2:],
                                    mode=self.interpolation,
                                    padding_mode=self.padding_mode)
        return sheared_img

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class ShearX(Shear):
    def __init__(self, *args, **kwargs):
        super(ShearX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class ShearY(Shear):
    def __init__(self, *args, **kwargs):
        super(ShearY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class Translate:
    def __init__(self, level,
                 randomness=True,
                 x_axis=True, y_axis=False,
                 padding_mode='zeros',
                 interpolation='bilinear',):
        self.level = level
        self.randomness = randomness
        self.x_axis, self.y_axis = x_axis, y_axis

        assert padding_mode in ['zeros', 'border', 'reflection'], \
            f"padding_mode should be 'zeros', 'border' or 'reflection', but got {padding_mode}"
        self.padding_mode = padding_mode

        assert interpolation in ['bilinear', 'nearest'], \
            f"interpolation should be 'bilinear' or 'nearest', but got {interpolation}"
        self.interpolation = interpolation

    def _get_translation(self, max_img_pixel):
        max_val = max_img_pixel / 3
        level = int(sample_level(self.level) * max_val / 10) \
            if self.randomness else int(self.level * max_val / 10)
        if np.random.uniform() > 0.5:
            level = -level
        return level

    def _get_translations(self, batch_size, max_img_pixel,
                                 fillzero=False, device='cuda'):
        if fillzero:
            return torch.zeros(batch_size, device=device)
        else:
            return torch.tensor(
                [self._get_translation(max_img_pixel)
                 for _ in range(batch_size)], device=device)

    def __call__(self, img, center=None, **kwargs):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        assert len(img.shape) == 4, f"img shape should be (bs, c, h, w), but got {img.shape}"

        bs = img.shape[0]
        device = img.get_device()

        translation_x = self._get_translations(
            bs, img.shape[-1],
            fillzero=(not self.x_axis), device=device)
        translation_y = self._get_translations(
            bs, img.shape[-2],
            fillzero=(not self.y_axis), device=device)
        translation = torch.stack([translation_x, translation_y], dim=-1)

        translated_img = T.translate(img,
                                     translation=translation,
                                     mode=self.interpolation,
                                     padding_mode=self.padding_mode)

        return translated_img

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class TranslateX(Translate):
    def __init__(self, *args, **kwargs):
        super(TranslateX, self).__init__(x_axis=True, y_axis=False, *args, **kwargs)


@TRANSFORMATIONS.register_module()
class TranslateY(Translate):
    def __init__(self, *args, **kwargs):
        super(TranslateY, self).__init__(x_axis=False, y_axis=True, *args, **kwargs)
