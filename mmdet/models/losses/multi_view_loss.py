# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_loss


@LOSSES.register_module()
class JSDLoss(nn.Module):
    def __init__(self, target, reduction='batchmean', loss_weight=1.0, name='loss_jsd'):
        """Jensen-Shannon Divergence (JSD) Loss. """
        super(JSDLoss, self).__init__()
        self.target = target
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.name = name

    def forward(self, data):
        """
        Args:
            data (dict[list[torch.Tensor]]): The data from hook.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert isinstance(data, dict), 'The data must be a dict.'
        data = data.get(self.target, None)
        assert data is not None, f'{self.target} is not in the data dict'
        if not isinstance(data, list): data = [data]
        for d in data:
            assert len(d.shape) == 2, 'The shape of data must be (bs, dim).'
        probs = [F.softmax(d, dim=1) for d in data]
        probs = [prob.reshape((1,) + prob.shape).contiguous() for prob in probs]
        prob_mixture = torch.clamp(torch.cat(probs, dim=0).mean(dim=0), 1e-7, 1).reshape(probs[0].shape).contiguous().log()

        loss = 0.0
        for prob in probs:
            loss += F.kl_div(prob_mixture, prob, reduction=self.reduction)
        loss /= len(probs)

        loss = self.loss_weight * loss

        return {self.name: loss}


@LOSSES.register_module()
class MultiViewLoss(nn.Module):

    def __init__(self, criterion):
        """ MultiViewLoss.

        Args:
            criterion (dict): The criterion dict.
        """
        super(MultiViewLoss, self).__init__()
        self.criterion = build_loss(criterion)

    def forward(self, hook_data):
        """Forward function.

        Args:
            hook_data (dict[list[torch.Tensor]): The data from hook.
        Returns:
            torch.Tensor: The calculated loss.
        """
        losses = dict()

        loss = self.criterion(hook_data)
        losses.update(loss)

        return losses
