from mmcv.utils import build_from_cfg
from mmcv.utils import Registry

DO_SOMETHING = Registry('do_something')

def build_do_something(cfg):
    return build_from_cfg(cfg, DO_SOMETHING)