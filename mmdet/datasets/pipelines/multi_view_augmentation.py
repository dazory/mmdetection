from ..builder import PIPELINES


@PIPELINES.register_module()
class MultiViewAugmentation:
    def __init__(self,
                 num_views=2,
                 use_clean=True,
                 ):
        super(MultiViewAugmentation, self).__init__()
        assert num_views > 0, f'num_views should be greater than 0, but got {num_views}'
        self.num_views = num_views
        self.use_clean = use_clean

    def __call__(self, results, *args, **kwargs):
        img = results.pop('img')
        gt_bboxes = results.pop('gt_bboxes')
        gt_labels = results.pop('gt_labels')
        for i in range(self.num_views):
            if self.use_clean and i == 0:
                aug_img, aug_gt_bboxes, aug_gt_labels = \
                    img.copy(), gt_bboxes.copy(), gt_labels.copy()
            else:
                aug_img, aug_gt_bboxes, aug_gt_labels = self.aug(
                    {'img': img.copy(),
                     'gt_bboxes': gt_bboxes.copy(),
                     'gt_labels': gt_labels.copy()})
            results[f'img{i}'] = aug_img
            results[f'gt_bboxes{i}'] = aug_gt_bboxes
            results[f'gt_labels{i}'] = aug_gt_labels

        results['img_fields'] = [f'img{i}' for i in range(self.num_views)]
        results['bbox_fields'] = [f'gt_bboxes{i}' for i in range(self.num_views)]
        results['label_fields'] = [f'gt_labels{i}' for i in range(self.num_views)]

        return results

    def aug(self, results):
        img, gt_bboxes, gt_labels = results['img'], results['gt_bboxes'], results['gt_labels']
        Warning('This function should be implemented in the child class.')
        '''
        Example:
            img = aug(img)
            # or gt_bboxes = aug(gt_bboxes)
            # or gt_labels = aug(gt_labels)
        '''
        return img, gt_bboxes, gt_labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
