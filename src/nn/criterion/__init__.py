"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register
from .det_criterion import DetCriterion
from .robust_loss import RobustLossWrapper
from .oadg_losses import (
    ContrastiveLossPlus, JSDLoss
)

CrossEntropyLoss = register()(nn.CrossEntropyLoss)
