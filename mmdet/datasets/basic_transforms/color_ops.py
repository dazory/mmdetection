import numpy as np
import kornia.augmentation as K

from ..builder import TRANSFORMATIONS


def sample_level(n):
    return np.random.uniform(0.1, n)


@TRANSFORMATIONS.register_module()
class AutoContrast:
    def __init__(self, p=1.0):
        self.func = K.RandomAutoContrast(p=p, same_on_batch=True)

    def __call__(self, img, *args, **kwargs):
        return self.func(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Equalize:
    def __init__(self, p=1.0):
        self.func = K.RandomEqualize(p=p, same_on_batch=True)

    def __call__(self, img, *args, **kwargs):
        return self.func(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Posterize:
    def __init__(self, p=1.0, level=4, randomness=True):
        self.p = p
        self.level = level
        self.max_val = 4
        self.randomness = randomness

    def _get_bits(self):
        bits = int(sample_level(self.level) * self.max_val / 10) \
            if self.randomness else int(self.level * self.max_val / 10)
        return bits

    def __call__(self, img, *args, **kwargs):
        bits = self._get_bits()
        return K.RandomPosterize(
            p=self.p, bits=self.max_val - bits, same_on_batch=True)(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Solarize:
    def __init__(self, level, p=1.0, randomness=True):
        self.level = level
        self.p = p
        self.randomness = randomness
        self.max_val = 256

    def _get_level(self):
        return int(sample_level(self.level) * self.max_val / 10) \
            if self.randomness else int(self.level * self.max_val / 10)

    def __call__(self, img, *args, **kwargs):
        level = self._get_level()
        return K.RandomSolarize(
            p=self.p, thresholds=self.max_val - level, same_on_batch=True)(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
