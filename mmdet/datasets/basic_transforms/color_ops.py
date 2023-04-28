import numpy as np
import kornia.augmentation as K

from ..builder import TRANSFORMATIONS


def sample_level(n):
    return np.random.uniform(0.1, n)


@TRANSFORMATIONS.register_module()
class AutoContrast:
    def __init__(self, p=1.0):
        self.func = K.RandomAutoContrast(p=p, same_on_batch=True)

    def __call__(self, img):
        return self.func(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Equalize:
    def __init__(self, p=1.0):
        self.func = K.RandomEqualize(p=p, same_on_batch=True)

    def __call__(self, img):
        return self.func(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Posterize:
    def __init__(self, p=1.0, level=4):
        max_val = 4
        bits = int(sample_level(level) * max_val / 10)
        self.func = K.RandomPosterize(p=p, bits=max_val - bits, same_on_batch=True)

    def __call__(self, img):
        return self.func(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMATIONS.register_module()
class Solarize:
    def __init__(self, level):
        max_val = 256
        level = int(sample_level(level) * max_val / 10)
        self.func = K.RandomSolarize(p=1.0,
                                     thresholds=max_val - level,
                                     same_on_batch=True)

    def __call__(self, img):
        return self.func(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
