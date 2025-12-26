from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        if self.downsample is None:
            return h, h
        return h, self.downsample(h)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample: bool,
    ):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2.0, mode="nearest") if upsample else None
        )
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Skip connection size mismatch")
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class Vis2IRFlowUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        if not channel_mults:
            raise ValueError("channel_mults must not be empty")
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        model_channels = [base_channels * mult for mult in channel_mults]
        in_ch = in_channels + cond_channels + 1

        self.downs = nn.ModuleList()
        for idx, out_ch in enumerate(model_channels):
            downsample = idx < len(model_channels) - 1
            self.downs.append(DownBlock(in_ch, out_ch, downsample))
            in_ch = out_ch

        self.mid = ConvBlock(in_ch, in_ch)

        self.ups = nn.ModuleList()
        for idx, skip_ch in enumerate(reversed(model_channels)):
            upsample = idx > 0
            out_ch = model_channels[-1] if idx == 0 else model_channels[-idx - 1]
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, upsample))
            in_ch = out_ch

        self.final = nn.Conv2d(in_ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        if cond.shape[-2:] != x.shape[-2:]:
            raise ValueError("cond and x must share spatial size")
        if cond.dim() != 4:
            raise ValueError("cond must be a 4D tensor")

        x_in = torch.cat([x, cond, t], dim=1)
        skips: list[torch.Tensor] = []
        h = x_in
        for down in self.downs:
            skip, h = down(h)
            skips.append(skip)

        h = self.mid(h)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)

        return self.final(h)
