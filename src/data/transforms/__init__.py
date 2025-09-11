"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._transforms import (
    ConvertBoxes,
    ConvertPILImage,
    EmptyTransform,
    Normalize,
    PadToSize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomIoUCrop,
    RandomPhotometricDistort,
    RandomZoomOut,
    Resize,
    ResizeWithPadding,
    SanitizeBoundingBoxes,
    # Enhanced transforms
    RandomBrightnessContrast,
    CLAHE,
    RandomGamma,
    BBoxMotionBlur,
    RandomRotate,  # Now points to SimpleRandomRotate
    SimpleRandomRotate,  # New implementation
    FixedRandomRotate,  # Backward compatibility alias
    GaussianNoise,
    JPEGCompression,
    RandomShadow,
    CoarseDropout,
    Transpose,
    GeometricAugmentationPipeline,
    CopyPaste,
    RandomErase,
)
from .container import Compose
from .mosaic import Mosaic
