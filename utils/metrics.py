from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    data_range_tensor = prediction.new_tensor(data_range)
    return 20.0 * torch.log10(data_range_tensor) - 10.0 * torch.log10(mse + eps)


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    if prediction.dim() != 4:
        raise ValueError("prediction and target must be 4D tensors")

    device = prediction.device
    dtype = prediction.dtype
    channels = prediction.shape[1]

    kernel = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    kernel = kernel.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    mu_x = F.conv2d(prediction, kernel, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.conv2d(prediction * prediction, kernel, padding=window_size // 2, groups=channels)
        - mu_x2
    )
    sigma_y2 = (
        F.conv2d(target * target, kernel, padding=window_size // 2, groups=channels)
        - mu_y2
    )
    sigma_xy = (
        F.conv2d(prediction * target, kernel, padding=window_size // 2, groups=channels)
        - mu_xy
    )

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))
