"""
D-FINE Hybrid Distortion-Aware Neck Module with Optional LWEGNet Integration
Maintains full compatibility with original HybridEncoder while adding optional enhancements
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from ...core import register
from .utils import get_activation

__all__ = ["HybridEncoder", "HybridDistortionAwareNeck", "create_enhanced_neck"]


# === LWEGNet Components (Optional Enhancement) ===

def build_norm_layer(norm_cfg, num_features):
    """Build normalization layer"""
    if isinstance(norm_cfg, dict):
        norm_type = norm_cfg.get("type", "BN")
    else:
        norm_type = norm_cfg or "BN"

    if norm_type == "BN":
        return "bn", nn.BatchNorm2d(num_features)
    elif norm_type == "SyncBN":
        return "bn", nn.SyncBatchNorm(num_features)
    elif norm_type == "GN":
        return "gn", nn.GroupNorm(32, num_features)
    else:
        return "bn", nn.BatchNorm2d(num_features)


class LWEGScharrModule(nn.Module):
    """LWEGNet Scharr edge detection module"""
    def __init__(self, channel, norm_layer="BN", act_layer=nn.ReLU):
        super().__init__()
        # Scharr filters
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return self.act(self.norm(scharr_edge))


class LWEGGaussianModule(nn.Module):
    """LWEGNet Gaussian filtering module"""
    def __init__(self, dim, size=5, sigma=1.0, norm_layer="BN", act_layer=nn.ReLU):
        super().__init__()
        gaussian = self.gaussian_kernel(size, sigma)
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=size//2, groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()

    def forward(self, x):
        gaussian_out = self.gaussian(x)
        return self.act(self.norm(gaussian_out))
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class LWEGLFEAModule(nn.Module):
    """LWEGNet Local Feature Enhancement and Attention module"""
    def __init__(self, channel, norm_layer="BN", act_layer=nn.ReLU):
        super().__init__()
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer()
        )
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
        return self.norm(c + att * wei)


# === Core Building Blocks (Maintain Full Compatibility) ===

class ConvNormLayer(nn.Module):
    """Standard convolution-normalization-activation layer"""
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    """Spatial-Channel Down-sampling"""
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """Standard VGG-style block"""
    def __init__(self, ch_in, ch_out, act="silu"):
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer"""
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", bottletype=VGGBlock):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class RepNCSPELAN4(nn.Module):
    """RepNCSPELAN4 block (maintain original structure for weight compatibility)"""
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        
        self.cv1 = ConvNormLayer(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer (maintain compatibility)"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


# === Optional Enhancement Modules ===

class EdgeEnhancementModule(nn.Module):
    """Unified edge enhancement module"""
    def __init__(self, channels: int, use_lweg: bool = False):
        super().__init__()
        self.use_lweg = use_lweg
        
        if use_lweg:
            self.scharr_detector = LWEGScharrModule(channels)
        
        # Standard Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        
        # Edge processing
        self.edge_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.edge_norm = nn.BatchNorm2d(channels)
        
        # Fusion
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False)
        self.fusion_norm = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sobel edge detection
        sobel_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        sobel_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        sobel_edges = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        if self.use_lweg:
            scharr_edges = self.scharr_detector(x)
            combined_edges = (sobel_edges + scharr_edges) / 2
        else:
            combined_edges = sobel_edges
        
        # Process and fuse
        edge_features = F.relu(self.edge_norm(self.edge_conv(combined_edges)))
        fused = torch.cat([x, edge_features], dim=1)
        return F.relu(self.fusion_norm(self.fusion_conv(fused)))


class BlurAwareModule(nn.Module):
    """Unified blur-aware enhancement module"""
    def __init__(self, channels: int, use_lweg: bool = False):
        super().__init__()
        self.use_lweg = use_lweg
        
        if use_lweg:
            self.gaussian_detector = LWEGGaussianModule(channels, size=7, sigma=2.0)
        
        # Standard Laplacian for blur detection
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))
        
        # Enhancement network
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels // 4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # Adaptive threshold
        self.threshold = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lweg:
            blur_response = self.gaussian_detector(x)
        else:
            # Standard blur detection
            gray = torch.mean(x, dim=1, keepdim=True)
            laplacian_response = F.conv2d(gray, self.laplacian, padding=1)
            blur_response = torch.var(laplacian_response, dim=[2, 3], keepdim=True).expand_as(x)
        
        threshold = self.threshold(x)
        blur_mask = (blur_response < threshold).float()
        enhanced = self.enhancer(x)
        
        return x + blur_mask * enhanced


@register()
class HybridDistortionAwareNeck(nn.Module):
    """
    Enhanced Hybrid Neck with Optional Distortion-Aware Processing
    
    Maintains 100% compatibility with original HybridEncoder weights
    while adding optional enhancement capabilities.
    """
    
    __share__ = ["eval_spatial_size"]
    
    def __init__(
        self,
        in_channels: List[int] = [512, 1024, 2048],
        feat_strides: List[int] = [8, 16, 32],
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = "gelu",
        use_encoder_idx: List[int] = [2],
        num_encoder_layers: int = 1,
        pe_temperature: int = 10000,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        act: str = "silu",
        eval_spatial_size: Optional[List[int]] = None,
        # Optional enhancement features (disabled by default)
        use_edge_enhancement: bool = False,
        use_blur_aware: bool = False,
        use_lweg_components: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        
        # Output configuration
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # === ORIGINAL STRUCTURE (100% compatible) ===
        
        # Input projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict([
                    ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ("norm", nn.BatchNorm2d(hidden_dim)),
                ])
            )
            self.input_proj.append(proj)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, enc_act, normalize_before=False
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])
        
        # FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2, hidden_dim, hidden_dim * 2,
                    round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act
                )
            )
        
        # PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(SCDown(hidden_dim, hidden_dim, 3, 2))
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2, hidden_dim, hidden_dim * 2,
                    round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act
                )
            )
        
        # === OPTIONAL ENHANCEMENTS ===
        
        if use_edge_enhancement:
            self.edge_enhancers = nn.ModuleList([
                EdgeEnhancementModule(hidden_dim, use_lweg_components) 
                for _ in range(len(in_channels))
            ])
        
        if use_blur_aware:
            self.blur_enhancers = nn.ModuleList([
                BlurAwareModule(hidden_dim, use_lweg_components) 
                for _ in range(len(in_channels))
            ])
        
        # Adaptive fusion for enhancements
        if use_edge_enhancement or use_blur_aware:
            if use_lweg_components:
                self.adaptive_fusion = nn.ModuleList([
                    LWEGLFEAModule(hidden_dim) for _ in range(len(in_channels))
                ])
            else:
                self.adaptive_fusion = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(hidden_dim, 1, 1),
                        nn.Sigmoid()
                    ) for _ in range(len(in_channels))
                ])

        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize position embeddings"""
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Build 2D sin-cos position embedding"""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
    
    def load_pretrained_weights(self, state_dict, strict=False):
        """Load pretrained weights with compatibility check"""
        compatible_dict = {}
        model_keys = set(self.state_dict().keys())
        
        for key, value in state_dict.items():
            if key in model_keys and self.state_dict()[key].shape == value.shape:
                compatible_dict[key] = value
        
        missing, unexpected = self.load_state_dict(compatible_dict, strict=False)
        print(f"Loaded {len(compatible_dict)} compatible layers")
        return missing, unexpected
    
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass with optional enhancements"""
        assert len(feats) == len(self.in_channels)
        
        # Input projection
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Optional enhancements
        if hasattr(self, 'edge_enhancers') or hasattr(self, 'blur_enhancers'):
            enhanced_feats = []
            for i, feat in enumerate(proj_feats):
                enhanced = feat
                
                if hasattr(self, 'edge_enhancers'):
                    edge_enhanced = self.edge_enhancers[i](enhanced)
                    if hasattr(self, 'adaptive_fusion'):
                        if isinstance(self.adaptive_fusion[i], LWEGLFEAModule):
                            enhanced = self.adaptive_fusion[i](enhanced, edge_enhanced)
                        else:
                            weight = self.adaptive_fusion[i](enhanced)
                            enhanced = enhanced * (1 - weight) + edge_enhanced * weight
                    else:
                        enhanced = edge_enhanced
                
                if hasattr(self, 'blur_enhancers'):
                    enhanced = self.blur_enhancers[i](enhanced)
                
                enhanced_feats.append(enhanced)
            proj_feats = enhanced_feats

        # Transformer encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


# === Factory Functions ===

def create_enhanced_neck(
    in_channels: List[int] = [512, 1024, 2048],
    feat_strides: List[int] = [8, 16, 32],
    hidden_dim: int = 256,
    enhancement_level: str = "none",  # "none", "basic", "full"
    **kwargs
) -> HybridDistortionAwareNeck:
    """
    Create neck with different enhancement levels
    
    Args:
        enhancement_level: 
            - "none": Original HybridEncoder (100% compatible)
            - "basic": Basic enhancements without LWEGNet
            - "full": Full enhancements with LWEGNet
    """
    
    if enhancement_level == "none":
        config = {
            'use_edge_enhancement': False,
            'use_blur_aware': False,
            'use_lweg_components': False,
        }
    elif enhancement_level == "basic":
        config = {
            'use_edge_enhancement': True,
            'use_blur_aware': True,
            'use_lweg_components': False,
        }
    elif enhancement_level == "full":
        config = {
            'use_edge_enhancement': True,
            'use_blur_aware': True,
            'use_lweg_components': True,
        }
    else:
        raise ValueError(f"Unknown enhancement_level: {enhancement_level}")
    
    config.update(kwargs)
    
    return HybridDistortionAwareNeck(
        in_channels=in_channels,
        feat_strides=feat_strides,
        hidden_dim=hidden_dim,
        **config
    )


# Backward compatibility
HybridEncoder = HybridDistortionAwareNeck

