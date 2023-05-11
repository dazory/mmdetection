# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
import PIL

from ..builder import PIPELINES

from .utils_deepaugment import EDSR, CAE, _get_weights_EDSR, _get_weights_CAE


def get_weights(model_name):
    if model_name == 'EDSR':
        return _get_weights_EDSR()
    elif model_name == 'CAE':
        return _get_weights_CAE()
    else:
        raise NotImplementedError


@PIPELINES.register_module()
class DeepAugment:
    def __init__(self, model_name, save_mode=False, use_flip=True):
        assert model_name in ['EDSR', 'CAE'], \
            f"model_name should be 'EDSR' or 'CAE', but got {model_name}"
        self.model_name = model_name
        self.net = EDSR() if model_name == 'EDSR' else CAE()
        weights = get_weights(model_name)
        self.net.load_state_dict(weights)
        self.net.cuda()
        self.net.eval()

        self.save_mode = save_mode
        self.use_flip = use_flip
        self.net.use_flip = use_flip

    def __call__(self, results):
        img = torch.Tensor(results['img'].copy()).permute(2, 0, 1)

        if np.random.uniform() < 0.05:
            weights = get_weights(model_name=self.model_name)
            self.net.load_state_dict(weights)
            self.net.eval()

        with torch.no_grad():
            if self.model_name == 'EDSR':
                pre_dist = set([random.randint(1, 4) for _ in range(1)])
                body_dist = set([random.randint(1, 5)])

                img = img.unsqueeze(0).cuda()
                img = F.interpolate(img, scale_factor=(1/4, 1/4))
                img = self.net(
                        img, pre_distortions=pre_dist, body_distortions=body_dist
                    )
                img = img.squeeze().clamp(0, 255)
            elif self.model_name == 'CAE':
                img = self.net(
                        img.unsqueeze(0).cuda() / 255.
                    ).squeeze().clamp(0, 1) * 255
            else:
                raise NotImplementedError

        img = img.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        if self.save_mode:
            img_prefix = results['img_prefix'].replace('/ws/data/cityscapes', f'/ws/data/cityscapes_{self.model_name}')
            file_name = results['img_info']['file_name']
            save_path = f"{img_prefix}{file_name}"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            img_vis = PIL.Image.fromarray(img)
            img_vis.save(save_path, quality='keep')  # If you want

        results['img'] = img
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.model_name})'


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,1)
# # img_vis = img.permute(1,2,0).cpu().detach().numpy()
# img_vis = img2.permute(1,2,0).cpu().detach().numpy()
# ax.imshow((img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()))
# # plt.savefig('/ws/data/dshong/mmdetection/deepaugment/0.orig.png')
# plt.savefig('/ws/data/dshong/mmdetection/deepaugment/1.aug.png')
# plt.close(fig)