from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def _pil_to_tensor(image) -> torch.Tensor:
    try:
        import numpy as np
    except ImportError:
        data = torch.tensor(list(image.getdata()), dtype=torch.float32)
        data = data.view(image.size[1], image.size[0])
        return data.unsqueeze(0) / 255.0
    array = np.array(image, dtype="float32")
    if array.ndim == 2:
        return torch.from_numpy(array).unsqueeze(0) / 255.0
    raise ValueError("Expected grayscale image")


def load_ir_sample(path: str, mode: Optional[str] = None):
    _ = mode
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
        return img.convert("L")


def paired_transform(vis, ir, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if not isinstance(ir, torch.Tensor):
        if Image is not None and isinstance(ir, Image.Image):
            if ir.size != (image_size, image_size):
                ir = ir.resize((image_size, image_size), Image.BILINEAR)
        ir_tensor = _pil_to_tensor(ir)
    else:
        ir_tensor = ir
        if ir_tensor.shape[-2:] != (image_size, image_size):
            ir_tensor = nn.functional.interpolate(
                ir_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    if not isinstance(vis, torch.Tensor):
        if Image is not None and isinstance(vis, Image.Image):
            if vis.size != (image_size, image_size):
                vis = vis.resize((image_size, image_size), Image.BILINEAR)
        vis_tensor = _pil_to_tensor(vis)
    else:
        vis_tensor = vis
        if vis_tensor.shape[-2:] != (image_size, image_size):
            vis_tensor = nn.functional.interpolate(
                vis_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    return vis_tensor, ir_tensor
