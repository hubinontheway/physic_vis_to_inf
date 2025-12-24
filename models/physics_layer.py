from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PhysicsLayer(nn.Module):
    """Differentiable physics layer for IR synthesis."""

    def __init__(self, use_mlp: bool = False, channels: int = 1) -> None:
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.g = nn.Sequential(
                nn.Conv2d(channels, 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, channels, kernel_size=1),
                nn.Softplus(),
            )
        else:
            self.g = nn.Softplus()

    def forward(
        self,
        temperature: torch.Tensor,
        emissivity: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g_t = self.g(temperature)
        ir_hat = emissivity * g_t
        if noise is not None:
            ir_hat = ir_hat + noise
        return ir_hat
