"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random

import torch
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor

torchvision.disable_beta_transforms_warning()


# @register()
# class Mosaic(T.Transform):
#     def __init__(
#         self,
#         size,
#         max_size=None,
#     ) -> None:
#         super().__init__()
#         self.resize = T.Resize(size=size, max_size=max_size)
#         self.crop = T.RandomCrop(size=max_size if max_size else size)

#         # TODO add arg `output_size` for affine`
#         # self.random_perspective = T.RandomPerspective(distortion_scale=0.5, p=1., )
#         self.random_affine = T.RandomAffine(
#             degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=114
#         )

#     def forward(self, *inputs):
#         inputs = inputs if len(inputs) > 1 else inputs[0]
#         image, target, dataset = inputs

#         images = []
#         targets = []
#         indices = random.choices(range(len(dataset)), k=3)
#         for i in indices:
#             image, target = dataset.load_item(i)
#             image, target = self.resize(image, target)
#             images.append(image)
#             targets.append(target)

#         # h, w = F.get_spatial_size(images[0])
#         w, h = images[0].size
#         offset = [[0, 0], [w, 0], [0, h], [w, h]]
#         image = Image.new(mode=images[0].mode, size=(w * 2, h * 2), color=0)
#         for i, im in enumerate(images):
#             image.paste(im, offset[i])

#         offset = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]]).repeat(1, 2)
#         target = {}
#         for k in targets[0]:
#             if k == "boxes":
#                 v = [t[k] + offset[i] for i, t in enumerate(targets)]
#             else:
#                 v = [t[k] for t in targets]

#             if isinstance(v[0], torch.Tensor):
#                 v = torch.cat(v, dim=0)

#             target[k] = v

#         if "boxes" in target:
#             # target['boxes'] = target['boxes'].clamp(0, 640 * 2 - 1)
#             w, h = image.size
#             target["boxes"] = convert_to_tv_tensor(
#                 target["boxes"], "boxes", box_format="xyxy", spatial_size=[h, w]
#             )

#         if "masks" in target:
#             target["masks"] = convert_to_tv_tensor(target["masks"], "masks")

#         image, target = self.random_affine(image, target)
#         # image, target = self.resize(image, target)
#         image, target = self.crop(image, target)

#         return image, target, dataset
    
@register()
class Mosaic(T.Transform):
    def __init__(
        self,
        size,
        max_size=None,
    ) -> None:
        super().__init__()
        self.resize = T.Resize(size=size, max_size=max_size)
        self.crop = T.RandomCrop(size=max_size if max_size else size)

        self.random_affine = T.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=114
        )

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs

        images = [image]
        targets = [target]
        indices = random.choices(range(len(dataset)), k=3)
        for i in indices:
            im, tgt = dataset.load_item(i)
            im, tgt = self.resize(im, tgt)
            images.append(im)
            targets.append(tgt)

        w, h = images[0].size
        offset = [[0, 0], [w, 0], [0, h], [w, h]]
        image = Image.new(mode=images[0].mode, size=(w * 2, h * 2), color=0)
        for i, im in enumerate(images):
            image.paste(im, offset[i])

        offset_tensor = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]]).repeat(1, 2)
        merged_target = {}
        for k in targets[0]:
            if k == "boxes":
                v = [t[k] + offset_tensor[i] for i, t in enumerate(targets)]
            elif k == "image_id":
                # only keep first image_id to avoid list for Path()
                v = targets[0][k]
            else:
                v = [t[k] for t in targets]

            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                v = torch.cat(v, dim=0)

            merged_target[k] = v

        if "boxes" in merged_target:
            target_boxes = convert_to_tv_tensor(
                merged_target["boxes"], "boxes", box_format="xyxy", spatial_size=[h * 2, w * 2]
            )
            merged_target["boxes"] = target_boxes

        if "masks" in merged_target:
            merged_target["masks"] = convert_to_tv_tensor(merged_target["masks"], "masks")

        if "image_id" in merged_target and isinstance(merged_target["image_id"], torch.Tensor):
            merged_target["image_id"] = merged_target["image_id"].unsqueeze(0)

        image, merged_target = self.random_affine(image, merged_target)
        image, merged_target = self.crop(image, merged_target)

        return image, merged_target, dataset
