# Copyright (c) OpenMMLab. All rights reserved.

from typing import List
from mmpl.registry import DATASETS

from mmseg.datasets.basesegdataset import BaseSegDataset
@DATASETS.register_module()
class UVSegDataset(BaseSegDataset):
    """UV SEG dataset.

    In segmentation map annotation for urban village, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = {
        'classes': ('background', 'UV',),
        'palette': [[120, 120, 120], [180, 120, 120],]
    }
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
