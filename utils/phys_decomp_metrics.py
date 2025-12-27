from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from losses.phys_losses import (
    consistency_loss,
    corr_loss,
    emissivity_prior_loss,
    recon_loss,
    smoothness_loss,
)


def compute_losses(
    v1: torch.Tensor,
    v2: torch.Tensor,
    t1: torch.Tensor,
    eps1: torch.Tensor,
    n1: Optional[torch.Tensor],
    ir1_hat: torch.Tensor,
    t2: torch.Tensor,
    eps2: torch.Tensor,
    n2: Optional[torch.Tensor],
    ir2_hat: torch.Tensor,
    weights: Dict[str, object],
) -> Dict[str, torch.Tensor]:
    loss_recon = recon_loss(ir1_hat, v1) + recon_loss(ir2_hat, v2)
    loss_t_smooth = smoothness_loss(t1) + smoothness_loss(t2)
    loss_eps_prior = emissivity_prior_loss(eps1) + emissivity_prior_loss(eps2)
    loss_consistency = consistency_loss(t1, t2, eps1, eps2, n1, n2)
    loss_corr = corr_loss(t1, eps1) + corr_loss(t2, eps2)

    total_loss = (
        float(weights.get("recon", 1.0)) * loss_recon
        + float(weights.get("t_smooth", 0.1)) * loss_t_smooth
        + float(weights.get("eps_prior", 0.1)) * loss_eps_prior
        + float(weights.get("consistency", 0.1)) * loss_consistency
        + float(weights.get("corr", 0.05)) * loss_corr
    )
    return {
        "total": total_loss,
        "recon": loss_recon,
        "t_smooth": loss_t_smooth,
        "eps_prior": loss_eps_prior,
        "consistency": loss_consistency,
        "corr": loss_corr,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: Dict[str, object],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        "total": 0.0,
        "recon": 0.0,
        "t_smooth": 0.0,
        "eps_prior": 0.0,
        "consistency": 0.0,
        "corr": 0.0,
    }
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict) or "ir" not in batch:
                raise ValueError("Expected dataset batch with 'ir' tensor")
            ir = batch["ir"].to(device)
            v1 = ir
            v2 = ir
            t1, eps1, n1, ir1_hat = model(v1)
            t2, eps2, n2, ir2_hat = model(v2)
            losses = compute_losses(
                v1,
                v2,
                t1,
                eps1,
                n1,
                ir1_hat,
                t2,
                eps2,
                n2,
                ir2_hat,
                weights,
            )
            batch_size = int(ir.shape[0])
            for key in totals:
                totals[key] += losses[key].item() * batch_size
            total_samples += batch_size
    if was_training:
        model.train()
    if total_samples == 0:
        raise ValueError("Evaluation loader returned no samples")
    return {key: value / total_samples for key, value in totals.items()}
