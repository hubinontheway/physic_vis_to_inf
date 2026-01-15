from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Dict

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Wrapper for torchmetrics.functional.peak_signal_noise_ratio.
    Maintains compatibility with the existing project signature.
    """
    # torchmetrics uses 'preds' and 'target'
    return peak_signal_noise_ratio(
        preds=prediction,
        target=target,
        data_range=data_range,
        dim=None, 
        reduction='elementwise_mean'
    )


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Wrapper for torchmetrics.functional.structural_similarity_index_measure.
    Maintains compatibility with the existing project signature.
    """
    return structural_similarity_index_measure(
        preds=prediction,
        target=target,
        data_range=data_range,
        kernel_size=window_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
        reduction='elementwise_mean'
    )


def _as_rgb(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError("Expected tensor with shape (B, C, H, W)")
    channels = tensor.shape[1]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError("Expected 1 or 3 channels for image metrics")


def _to_255(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 255.0).clamp(0.0, 255.0)


def _to_lpips_range(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 - 1.0


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


@dataclass
class PerceptualMetrics:
    fid: FrechetInceptionDistance
    kid: KernelInceptionDistance
    lpips: LearnedPerceptualImagePatchSimilarity
    lpips_sum: float = 0.0
    sample_count: int = 0

    @classmethod
    def create(cls, device: torch.device, lpips_net: str = "alex") -> "PerceptualMetrics":
        fid = FrechetInceptionDistance().to(device)
        kid = KernelInceptionDistance().to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net).to(device)
        fid.eval()
        kid.eval()
        lpips.eval()
        return cls(fid=fid, kid=kid, lpips=lpips)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        pred_rgb = _as_rgb(prediction)
        target_rgb = _as_rgb(target)

        fid_pred = _to_255(pred_rgb)
        fid_target = _to_255(target_rgb)
        self.fid.update(fid_target, real=True)
        self.fid.update(fid_pred, real=False)
        self.kid.update(fid_target, real=True)
        self.kid.update(fid_pred, real=False)

        lpips_pred = _to_lpips_range(pred_rgb)
        lpips_target = _to_lpips_range(target_rgb)
        lpips_val = self.lpips(lpips_pred, lpips_target)
        batch = int(pred_rgb.shape[0])
        if lpips_val.dim() == 0:
            self.lpips_sum += _as_float(lpips_val) * batch
        else:
            self.lpips_sum += float(lpips_val.sum().item())
        self.sample_count += batch

    def compute(self) -> Dict[str, float]:
        fid_val = _as_float(self.fid.compute())
        kid_val = self.kid.compute()
        kid_mean: float
        kid_std: float | None = None
        if isinstance(kid_val, tuple):
            kid_mean = _as_float(kid_val[0])
            kid_std = _as_float(kid_val[1])
        elif isinstance(kid_val, torch.Tensor) and kid_val.numel() == 2:
            kid_mean = _as_float(kid_val[0])
            kid_std = _as_float(kid_val[1])
        else:
            kid_mean = _as_float(kid_val)

        lpips_mean = self.lpips_sum / max(self.sample_count, 1)
        metrics = {"fid": fid_val, "kid": kid_mean, "lpips": lpips_mean}
        if kid_std is not None:
            metrics["kid_std"] = kid_std
        return metrics
