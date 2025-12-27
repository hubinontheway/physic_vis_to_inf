from __future__ import annotations

from typing import Dict

import torch

from models.phys_decomp import PhysDecompNet
from models.vit_unet_phys import PhysDecompViTUNet


def _ensure_timm_available() -> None:
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "timm is required for model_type=vit_unet with use_timm=true. "
            "Install with `pip install timm`."
        ) from exc


def create_phys_decomp_model(
    config: Dict[str, object],
    image_size: int,
    t_min: float,
    t_max: float,
    use_noise: bool,
    device: torch.device,
) -> torch.nn.Module:
    model_type = str(config.get("model_type", "cnn")).lower()
    if model_type == "vit_unet":
        vit_cfg = config.get("vit_unet", {})
        if not isinstance(vit_cfg, dict):
            raise ValueError("'vit_unet' must be a mapping")
        if bool(vit_cfg.get("use_timm", False)):
            _ensure_timm_available()
        return PhysDecompViTUNet(
            in_channels=1,
            base_channels=int(vit_cfg.get("base_channels", 32)),
            vit_embed_dim=int(vit_cfg.get("vit_embed_dim", 256)),
            vit_depth=int(vit_cfg.get("vit_depth", 6)),
            vit_heads=int(vit_cfg.get("vit_heads", 8)),
            patch_size=int(vit_cfg.get("patch_size", 16)),
            image_size=image_size,
            t_min=t_min,
            t_max=t_max,
            use_noise=use_noise,
            use_mlp=bool(vit_cfg.get("use_mlp", False)),
            use_timm=bool(vit_cfg.get("use_timm", False)),
            timm_model=str(vit_cfg.get("timm_model", "vit_base_patch16_224")),
            timm_pretrained=bool(vit_cfg.get("timm_pretrained", False)),
        ).to(device)
    return PhysDecompNet(
        in_channels=1,
        base_channels=32,
        t_min=t_min,
        t_max=t_max,
        use_noise=use_noise,
    ).to(device)
