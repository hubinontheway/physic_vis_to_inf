from __future__ import annotations

import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


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