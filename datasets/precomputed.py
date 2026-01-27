from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.vision import load_tensor_or_pil


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


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    if tensor.numel() == 0:
        return tensor
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.max() > 1.0 or tensor.min() < 0.0:
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


class PrecomputedIR0Dataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        ir0_root: str,
        image_size: Optional[int] = None,
        mode: str = "L",
        use_split_subdir: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.ir0_root = ir0_root
        self.image_size = image_size
        self.mode = mode
        split = getattr(base_dataset, "split", None)
        if use_split_subdir and split:
            self.ir0_dir = os.path.join(ir0_root, str(split))
        else:
            self.ir0_dir = ir0_root

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _load_ir0(self, path: str) -> torch.Tensor:
        data = load_tensor_or_pil(path, mode=self.mode)
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = _pil_to_tensor(data)
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected a single image in {path}")
            tensor = tensor.squeeze(0)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"Expected image tensor with shape (C, H, W) for {path}")
        tensor = _normalize_tensor(tensor)
        if tensor.shape[0] != 1:
            tensor = tensor[:1]
        if self.image_size is not None and tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        ir_path = sample.get("ir_path")
        if not ir_path:
            raise ValueError("Base dataset must provide 'ir_path' for precomputed IR0 lookup")
        filename = os.path.basename(ir_path)
        ir0_path = os.path.join(self.ir0_dir, filename)
        if not os.path.exists(ir0_path):
            raise FileNotFoundError(f"Precomputed IR0 not found: {ir0_path}")
        sample["ir0"] = self._load_ir0(ir0_path)
        sample["ir0_path"] = ir0_path
        return sample
