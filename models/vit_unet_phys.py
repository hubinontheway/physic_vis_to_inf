from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from .physics_layer import PhysicsLayer

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


class PatchEmbedding(nn.Module):
    """Image to patch embedding with optional position encoding."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed: Optional[nn.Parameter] = None

    def _init_pos_embed(self, num_patches: int, device: torch.device) -> None:
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.proj.out_channels, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        self._init_pos_embed(tokens.shape[1], tokens.device)
        tokens = tokens + self.pos_embed
        return tokens, (h, w)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder that outputs a spatial feature map."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 16,
        image_size: Optional[int] = None,
        dropout: float = 0.0,
        use_timm: bool = False,
        timm_model: str = "vit_base_patch16_224",
        timm_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.use_timm = use_timm
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        if use_timm:
            if timm is None:
                raise ImportError("timm is required for use_timm=True")
            timm_kwargs = {
                "pretrained": timm_pretrained,
                "num_classes": 0,
                "in_chans": in_channels,
            }
            if image_size is not None:
                timm_kwargs["img_size"] = int(image_size)
            self.timm_model = timm.create_model(timm_model, **timm_kwargs)
            self.embed_dim = self.timm_model.num_features
            patch = self.timm_model.patch_embed.patch_size
            self.patch_size = int(patch[0] if isinstance(patch, tuple) else patch)
        else:
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_timm:
            tokens = self.timm_model.forward_features(x)
            if tokens.dim() == 3:
                h = x.shape[2] // self.patch_size
                w = x.shape[3] // self.patch_size
                num_patches = h * w
                if tokens.shape[1] == num_patches + 1:
                    tokens = tokens[:, 1:, :]
                elif tokens.shape[1] != num_patches:
                    raise ValueError("Unexpected timm token count for input size")
                return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)
            if tokens.dim() == 4:
                return tokens
            raise ValueError("Unexpected timm ViT feature shape")

        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError("Input size must be divisible by patch_size")
        tokens, (h, w) = self.patch_embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)


class ConvStem(nn.Module):
    """CNN pyramid to provide multi-scale skips aligned with ViT patch grid."""

    def __init__(self, in_channels: int, base_channels: int, num_stages: int) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        in_ch = base_channels
        for idx in range(1, num_stages + 1):
            out_ch = base_channels * (2**idx)
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up_blocks = nn.ModuleList()
        current_channels = out_channels
        for skip_ch in skip_channels:
            next_channels = max(out_channels // 2, 16)
            self.up_blocks.append(UpBlock(current_channels, skip_ch, next_channels))
            current_channels = next_channels
            out_channels = next_channels

        self.out_channels = current_channels

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = self.bottleneck(x)
        for block, skip in zip(self.up_blocks, skips):
            x = block(x, skip)
        return x


class PhysDecompViTUNet(nn.Module):
    """ViT encoder + U-Net decoder for physical property decoupling."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        vit_embed_dim: int = 256,
        vit_depth: int = 6,
        vit_heads: int = 8,
        patch_size: int = 16,
        image_size: Optional[int] = None,
        t_min: float = 1.0,
        t_max: float = 10.0,
        use_noise: bool = True,
        use_mlp: bool = False,
        use_timm: bool = False,
        timm_model: str = "vit_base_patch16_224",
        timm_pretrained: bool = False,
    ) -> None:
        super().__init__()
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")
        if patch_size <= 0 or patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of 2")

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.use_noise = use_noise
        num_stages = int(math.log2(patch_size))

        self.stem = ConvStem(in_channels, base_channels, num_stages)
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            patch_size=patch_size,
            image_size=image_size,
            use_timm=use_timm,
            timm_model=timm_model,
            timm_pretrained=timm_pretrained,
        )

        stem_channels = [base_channels * (2**idx) for idx in range(num_stages + 1)]
        bottleneck_channels = stem_channels[-1] + self.encoder.embed_dim
        skip_channels = list(reversed(stem_channels[:-1]))
        decoder_out = max(base_channels * 4, 32)
        self.decoder = UNetDecoder(
            in_channels=bottleneck_channels,
            skip_channels=skip_channels,
            out_channels=decoder_out,
        )

        self.head_t = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)
        self.head_eps = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)
        self.head_noise = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)
        self.physics = PhysicsLayer(use_mlp=use_mlp, channels=1)

    def _scale_temperature(self, raw: torch.Tensor) -> torch.Tensor:
        scaled = torch.sigmoid(raw)
        return scaled * (self.t_max - self.t_min) + self.t_min

    def forward(
        self, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        skips = self.stem(ir)
        vit_feat = self.encoder(ir)
        if vit_feat.shape[-2:] != skips[-1].shape[-2:]:
            raise ValueError("ViT feature map resolution must match stem bottleneck")
        bottleneck = torch.cat([vit_feat, skips[-1]], dim=1)

        decoder_skips = list(reversed(skips[:-1]))
        features = self.decoder(bottleneck, decoder_skips)

        t_raw = self.head_t(features)
        eps_raw = self.head_eps(features)
        noise = self.head_noise(features) if self.use_noise else None

        temperature = self._scale_temperature(t_raw)
        emissivity = torch.sigmoid(eps_raw)
        ir_hat = self.physics(temperature, emissivity, noise)
        return temperature, emissivity, noise, ir_hat
