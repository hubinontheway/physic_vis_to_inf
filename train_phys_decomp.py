from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from losses.phys_losses import (
    consistency_loss,
    corr_loss,
    emissivity_prior_loss,
    recon_loss,
    smoothness_loss,
)
from datasets import create_dataset
from models.phys_decomp import PhysDecompNet
from models.vit_unet_phys import PhysDecompViTUNet
from utils.config import load_yaml

def _sample_geom_params(tensor: torch.Tensor, size: int) -> Tuple[int, int, bool, bool]:
    _, h, w = tensor.shape
    if h < size or w < size:
        raise ValueError("Image size must be >= crop size")
    if h == size and w == size:
        top = 0
        left = 0
    else:
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    return top, left, flip_h, flip_v


def _apply_geom(
    tensor: torch.Tensor, size: int, top: int, left: int, flip_h: bool, flip_v: bool
) -> torch.Tensor:
    view = tensor[:, top : top + size, left : left + size]
    if flip_h:
        view = torch.flip(view, dims=[2])
    if flip_v:
        view = torch.flip(view, dims=[1])
    return view


def _augment_photometric(ir: torch.Tensor) -> torch.Tensor:
    view = ir
    gain = random.uniform(0.8, 1.2)
    bias = random.uniform(-0.1, 0.1)
    gamma = random.uniform(0.8, 1.2)
    view = torch.clamp(view * gain + bias, 0.0, 1.0)
    view = torch.pow(view, gamma)
    noise = torch.randn_like(view) * 0.02
    view = torch.clamp(view + noise, 0.0, 1.0)
    return view


def _prepare_batch(ir: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    v1_list = []
    v2_list = []
    for sample in ir:
        top, left, flip_h, flip_v = _sample_geom_params(sample, size)
        base = _apply_geom(sample, size, top, left, flip_h, flip_v)
        v1_list.append(_augment_photometric(base))
        v2_list.append(_augment_photometric(base))
    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)
    return v1, v2


def _save_visualization(
    output_dir: str,
    step: int,
    ir: torch.Tensor,
    ir_hat: torch.Tensor,
    temperature: torch.Tensor,
    emissivity: torch.Tensor,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        torch.save(
            {
                "ir": ir.cpu(),
                "ir_hat": ir_hat.cpu(),
                "temperature": temperature.cpu(),
                "emissivity": emissivity.cpu(),
            },
            os.path.join(output_dir, f"step_{step}.pt"),
        )
        return

    def _to_pil(tensor: torch.Tensor) -> Image.Image:
        data = tensor.detach().cpu()
        data = data - data.min()
        data = data / (data.max() + 1e-6)
        data = (data * 255.0).byte().squeeze(0)
        return Image.fromarray(data.numpy(), mode="L")

    images = [
        _to_pil(ir[0]),
        _to_pil(ir_hat[0]),
        _to_pil(temperature[0]),
        _to_pil(emissivity[0]),
    ]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    grid = Image.new("L", (total_width, max_height))
    offset = 0
    for img in images:
        grid.paste(img, (offset, 0))
        offset += img.size[0]
    grid.save(os.path.join(output_dir, f"step_{step}.png"))


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


def _load_ir_sample(path: str, mode: Optional[str] = None):
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


def _paired_transform(vis, ir, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from PIL import Image
    except ImportError:
        Image = None

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


def _validate_config(config: Dict[str, object]) -> None:
    required = ["lr", "batch_size", "image_size", "loss_weights"]
    for key in required:
        if key not in config:
            raise KeyError(f"train config missing '{key}'")
    if not isinstance(config["loss_weights"], dict):
        raise ValueError("'loss_weights' must be a mapping")
    if "dataset" not in config:
        raise KeyError("train config missing 'dataset'")


def _ensure_timm_available() -> None:
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "timm is required for model_type=vit_unet with use_timm=true. "
            "Install with `pip install timm`."
        ) from exc


def run_training(config_path: str, steps_override: int | None = None) -> None:
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("train config must be a mapping")
    _validate_config(config)

    lr = float(config["lr"])
    batch_size = int(config["batch_size"])
    image_size = int(config["image_size"])
    steps = int(config.get("steps", 100))
    if steps_override is not None:
        steps = int(steps_override)
    vis_interval = int(config.get("vis_interval", 50))
    use_noise = bool(config.get("use_noise", True))
    t_min = float(config.get("t_min", 1.0))
    t_max = float(config.get("t_max", 10.0))
    seed = int(config.get("seed", 123))

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    split = str(config.get("split", "train"))
    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=split,
        loader=_load_ir_sample,
        transform=lambda v, r: _paired_transform(v, r, image_size),
        vis_mode="L",
        ir_mode="L",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    iterator: Iterator[torch.Tensor] = iter(loader)

    model_type = str(config.get("model_type", "cnn")).lower()
    if model_type == "vit_unet":
        vit_cfg = config.get("vit_unet", {})
        if not isinstance(vit_cfg, dict):
            raise ValueError("'vit_unet' must be a mapping")
        if bool(vit_cfg.get("use_timm", False)):
            _ensure_timm_available()
        model = PhysDecompViTUNet(
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
    else:
        model = PhysDecompNet(
            in_channels=1,
            base_channels=32,
            t_min=t_min,
            t_max=t_max,
            use_noise=use_noise,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights = config["loss_weights"]
    runs_dir = os.path.join("runs", "phys_decomp")

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        if not isinstance(batch, dict) or "ir" not in batch:
            raise ValueError("Expected dataset batch with 'ir' tensor")
        ir = batch["ir"].to(device)
        v1, v2 = _prepare_batch(ir, image_size)
        v1 = v1.to(device)
        v2 = v2.to(device)

        t1, eps1, n1, ir1_hat = model(v1)
        t2, eps2, n2, ir2_hat = model(v2)

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

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            "step={step} total={total:.4f} recon={recon:.4f} t_smooth={t_smooth:.4f} "
            "eps_prior={eps_prior:.4f} consistency={consistency:.4f} corr={corr:.4f}".format(
                step=step,
                total=total_loss.item(),
                recon=loss_recon.item(),
                t_smooth=loss_t_smooth.item(),
                eps_prior=loss_eps_prior.item(),
                consistency=loss_consistency.item(),
                corr=loss_corr.item(),
            )
        )

        if vis_interval > 0 and step % vis_interval == 0:
            _save_visualization(runs_dir, step, v1, ir1_hat, t1, eps1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PhysDecompNet training")
    parser.add_argument(
        "--config",
        default="configs/phys_decomp.yml",
        help="Path to training config",
    )
    parser.add_argument("--steps", type=int, help="Override number of steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config, steps_override=args.steps)


if __name__ == "__main__":
    main()
