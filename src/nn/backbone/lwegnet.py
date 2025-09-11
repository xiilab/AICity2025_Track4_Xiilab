# unified_backbone_detectron2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
import math
import os
import copy
import numpy
import matplotlib.pyplot as plt
from detectron2.modeling.backbone import get_norm, Backbone
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec, DropPath


def build_norm_layer(norm_cfg, num_features):
    if isinstance(norm_cfg, dict):
        norm_type = norm_cfg.get("type", "BN")
    else:
        norm_type = norm_cfg

    if norm_type == "BN":
        return "bn", nn.BatchNorm2d(num_features)
    elif norm_type == "SyncBN":
        return "bn", nn.SyncBatchNorm(num_features)
    elif norm_type == "GN":
        return "gn", nn.GroupNorm(32, num_features)
    elif norm_type == "FrozenBN":
        from detectron2.layers.batch_norm import FrozenBatchNorm2d
        return "bn", FrozenBatchNorm2d(num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


class UnifiedBackbone(Backbone):
    def __init__(self, 
                 backbone_type="lwegnet",
                 lweg_cfg=None,
                 hgnet_cfg=None):
        super().__init__()

        if backbone_type == "lwegnet":
            self.backbone = LWEGNet(**(lweg_cfg or {}))
            self.out_channels = [
                int(lweg_cfg.get("stem_dim", 32) * 2 ** i) for i in range(len(self.backbone.out_indices))
            ]
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def forward(self, x):
        features = self.backbone(x)
        return {f"res{i+2}": feat for i, feat in enumerate(features)}

    def output_shape(self):
        return {
            f"res{i+2}": ShapeSpec(
                channels=self.out_channels[i],
                stride=2 ** (i + 2)
            ) for i in range(len(self.out_channels))
        }


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out


class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        # 정의 Scharr 필터
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # Sobel 필터를 컨볼루션 레이어에 할당
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        # 컨볼루션 연산 적용
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # 엣지와 가우시안 분포 강도 계산
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)

        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)

        return x


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


class LFE_Module(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim, norm_layer, act_layer)

        if stage == 0:
            self.Scharr_edge = Scharr(dim, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor) -> Tensor:
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):
        super().__init__()

        blocks_list = [
            LFE_Module(
                dim=dim,
                stage=stage,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        """LoG 커널 생성"""
        # 초기화 2D 좌표
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        # LoG 커널 계산
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 정규화
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0) # 배치와 채널 차원 추가
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]
    
    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x
    

class Stem(nn.Module):

    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 2
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1])
        # LoG 필터 정의
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        # gaussian
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd = DRFD(out_c12, norm_layer, act_layer)

    def forward(self, x):
        x = self.LoG(x)
        # original size to 2x downsampling layer
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)

        return x  # x = [B, C, H/4, W/4]


class LWEGNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer="BN",
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 fork_feat=True):
        super().__init__()

        self.num_stages = len(depths)
        self.out_channels = [int(stem_dim * 2 ** i) for i in range(self.num_stages)]

        self.stem = Stem(
            in_chans=in_chans,
            stem_dim=stem_dim,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        stages = []
        self.out_indices = []
        for i in range(self.num_stages):
            dim = int(stem_dim * 2 ** i)
            stage = BasicStage(
                dim=dim,
                stage=i,
                depth=depths[i],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            stages.append(stage)
            setattr(self, f"stage_{i}", stage)
            if i < self.num_stages - 1:
                drfd = DRFD(dim=dim, norm_layer=norm_layer, act_layer=act_layer)
                stages.append(drfd)
                setattr(self, f"drfd_{i}", drfd)
            self.out_indices.append(len(stages) - 1)

        self.stages = nn.Sequential(*stages)

        for i, idx in enumerate(self.out_indices):
            _, layer = build_norm_layer(norm_layer, self.out_channels[i])
            setattr(self, f'norm{idx}', layer)

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                outputs.append(norm_layer(x))
        return tuple(outputs)


@BACKBONE_REGISTRY.register()
def build_unified_backbone(cfg, input_shape):
    backbone_type = cfg.MODEL.BACKBONE.NAME
    lweg_cfg = {}
    
    if hasattr(cfg.MODEL, "LWEG"):
        lweg_cfg = {
            "in_chans": input_shape.channels,
            "stem_dim": cfg.MODEL.LWEG.STEM_DIM if hasattr(cfg.MODEL.LWEG, "STEM_DIM") else 32,
            "depths": cfg.MODEL.LWEG.DEPTHS if hasattr(cfg.MODEL.LWEG, "DEPTHS") else (1, 4, 4, 2),
            "norm_layer": cfg.MODEL.LWEG.NORM if hasattr(cfg.MODEL.LWEG, "NORM") else "BN",
            "drop_path_rate": cfg.MODEL.LWEG.DROP_PATH_RATE if hasattr(cfg.MODEL.LWEG, "DROP_PATH_RATE") else 0.1,
            "mlp_ratio": cfg.MODEL.LWEG.MLP_RATIO if hasattr(cfg.MODEL.LWEG, "MLP_RATIO") else 2.0,
        }

    model = UnifiedBackbone(
        backbone_type=backbone_type,
        lweg_cfg=lweg_cfg
    )
    return model
