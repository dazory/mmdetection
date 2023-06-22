# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS

from .wandblogger_hook import MMDetWandbHook


@HOOKS.register_module()
class CustomMMDetWandbHook(MMDetWandbHook):
    def __init__(self, *args, **kwargs):
        super(CustomMMDetWandbHook, self).__init__(*args, **kwargs)

    # for the reason of this double-layered structure, refer to
    # https://github.com/open-mmlab/mmdetection/issues/8145#issuecomment-1345343076
    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super(MMDetWandbHook, self).after_train_iter(runner)
        else:
            super(MMDetWandbHook, self).after_train_iter(runner)
        self._after_train_iter(runner)
        if self.every_n_inner_iters(runner, self.interval):
            for k, v in runner.outputs.items():
                if 'scale' in k:
                    self.wandb.log({k: v})
                if 'dist' in k:
                    self.wandb.log({k: v})
