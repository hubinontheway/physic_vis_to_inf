from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def recon_loss(ir_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(ir_hat - target))


def smoothness_loss(tensor: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
    dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)


def emissivity_prior_loss(eps: torch.Tensor) -> torch.Tensor:
    # Encourage piecewise smooth emissivity with mild sparsity.
    tv = smoothness_loss(eps)
    sparse = torch.mean(torch.abs(eps))
    return tv + 0.1 * sparse


def consistency_loss(
    t1: torch.Tensor,
    t2: torch.Tensor,
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    n1: Optional[torch.Tensor] = None,
    n2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = torch.mean(torch.abs(t1 - t2)) + torch.mean(torch.abs(eps1 - eps2))
    if n1 is not None and n2 is not None:
        loss = loss + torch.mean(torch.abs(n1 - n2))
    return loss


def corr_loss(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    flat_a = tensor_a.flatten(start_dim=1)
    flat_b = tensor_b.flatten(start_dim=1)
    flat_a = flat_a - flat_a.mean(dim=1, keepdim=True)
    flat_b = flat_b - flat_b.mean(dim=1, keepdim=True)
    denom = (
        torch.sqrt(torch.sum(flat_a ** 2, dim=1))
        * torch.sqrt(torch.sum(flat_b ** 2, dim=1))
        + 1e-6
    )
    corr = torch.sum(flat_a * flat_b, dim=1) / denom
    return torch.mean(torch.abs(corr))


class PhysDecompLoss(nn.Module):
    """Combined reconstruction + consistency loss for two-view training."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        consistency_weight: float = 0.1,
        recon_type: str = "l1",
    ) -> None:
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.consistency_weight = float(consistency_weight)
        if recon_type not in {"l1", "l2"}:
            raise ValueError("recon_type must be 'l1' or 'l2'")
        self.recon_type = recon_type

    def _recon(self, ir_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_type == "l2":
            return F.mse_loss(ir_hat, target)
        return recon_loss(ir_hat, target)

    def forward(
        self,
        ir_hat1: torch.Tensor,
        ir_hat2: torch.Tensor,
        ir1: torch.Tensor,
        ir2: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        eps1: torch.Tensor,
        eps2: torch.Tensor,
        n1: Optional[torch.Tensor] = None,
        n2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        recon = self._recon(ir_hat1, ir1) + self._recon(ir_hat2, ir2)
        consistency = consistency_loss(t1, t2, eps1, eps2, n1, n2)
        total = self.recon_weight * recon + self.consistency_weight * consistency
        return {"total": total, "recon": recon, "consistency": consistency}
