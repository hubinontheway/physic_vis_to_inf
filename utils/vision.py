from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def _pil_to_tensor(image) -> torch.Tensor:
    try:
        import numpy as np
    except ImportError:
        data = list(image.getdata())
        if image.mode == "RGB":
            tensor = torch.tensor(data, dtype=torch.float32).view(
                image.size[1], image.size[0], 3
            )
            tensor = tensor.permute(2, 0, 1)
            return tensor / 255.0
        tensor = torch.tensor(data, dtype=torch.float32).view(
            image.size[1], image.size[0]
        )
        return tensor.unsqueeze(0) / 255.0

    array = np.array(image, dtype="float32")
    if array.ndim == 2:
        return torch.from_numpy(array).unsqueeze(0) / 255.0
    if array.ndim == 3:
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0
    raise ValueError("Unsupported image array shape")


def tensor_to_pil(tensor: torch.Tensor, mode: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image saving") from exc
    data = tensor.detach().cpu()
    data = torch.clamp(data, 0.0, 1.0)
    if mode == "RGB":
        data = (data * 255.0).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(data, mode="RGB")
    data = (data * 255.0).byte().squeeze(0).numpy()
    return Image.fromarray(data, mode="L")


def load_tensor_or_pil(path: str, mode: Optional[str] = None):
    if path.endswith(".pt"):
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}")
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.float()
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load image files") from exc
    with Image.open(path) as img:
        if mode:
            img = img.convert(mode)
        return img


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    if tensor.numel() > 0 and tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.numel() > 0 and (tensor.max() > 1.0 or tensor.min() < 0.0):
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def _ensure_vis_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 3:
        raise ValueError("Expected vis tensor to have shape (C, H, W)")
    if tensor.shape[0] == 3:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    raise ValueError("Visible tensor must have 1 or 3 channels")


def paired_transform(
    vis,
    ir,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(vis, torch.Tensor):
        vis_tensor = _pil_to_tensor(vis)
    else:
        vis_tensor = vis
    if not isinstance(ir, torch.Tensor):
        ir_tensor = _pil_to_tensor(ir)
    else:
        ir_tensor = ir

    vis_tensor = _normalize_tensor(vis_tensor)
    ir_tensor = _normalize_tensor(ir_tensor)

    vis_tensor = _ensure_vis_channels(vis_tensor)
    if ir_tensor.dim() == 2:
        ir_tensor = ir_tensor.unsqueeze(0)
    if ir_tensor.shape[0] != 1:
        ir_tensor = ir_tensor[:1]

    if vis_tensor.shape[-2:] != (image_size, image_size):
        vis_tensor = nn.functional.interpolate(
            vis_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if ir_tensor.shape[-2:] != (image_size, image_size):
        ir_tensor = nn.functional.interpolate(
            ir_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return vis_tensor, ir_tensor
