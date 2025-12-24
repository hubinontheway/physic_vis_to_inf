from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .physics_layer import PhysicsLayer


class PhysDecompNet(nn.Module):
    """Physically inspired decomposition network."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        t_min: float = 1.0,
        t_max: float = 10.0,
        use_noise: bool = True,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.use_noise = use_noise

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head_t = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.head_eps = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.head_noise = nn.Conv2d(base_channels, 1, kernel_size=1)

        self.physics = PhysicsLayer(use_mlp=use_mlp, channels=1)

    def _scale_temperature(self, raw: torch.Tensor) -> torch.Tensor:
        scaled = torch.sigmoid(raw)
        return scaled * (self.t_max - self.t_min) + self.t_min

    def forward(
        self, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        features = self.encoder(ir)
        t_raw = self.head_t(features)
        eps_raw = self.head_eps(features)
        noise = self.head_noise(features) if self.use_noise else None

        temperature = self._scale_temperature(t_raw)
        emissivity = torch.sigmoid(eps_raw)
        ir_hat = self.physics(temperature, emissivity, noise)

        return temperature, emissivity, noise, ir_hat
