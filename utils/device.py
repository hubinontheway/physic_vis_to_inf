from __future__ import annotations

from typing import Dict

import torch


def resolve_device(config: Dict[str, object]) -> torch.device:
    device_value = config.get("device")
    if device_value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device_value, int):
        device_str = f"cuda:{device_value}"
    else:
        device_str = str(device_value)

    try:
        device = torch.device(device_str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid device '{device_value}'") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available")
        if device.index is not None:
            count = torch.cuda.device_count()
            if device.index < 0 or device.index >= count:
                raise ValueError(
                    f"CUDA device index {device.index} out of range "
                    f"(device_count={count})"
                )
            torch.cuda.set_device(device.index)
    return device
