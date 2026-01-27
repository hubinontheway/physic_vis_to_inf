from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.metrics import PerceptualMetrics, psnr, ssim
from utils.config import load_yaml
from utils.vision import load_tensor_or_pil

DEFAULT_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".pt",
}


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

    try:
        image.load()
    except Exception:
        pass
    try:
        array = np.array(image, dtype="float32")
    except Exception:
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
    if tensor.min() < 0.0 and tensor.max() <= 1.0 and tensor.min() >= -1.0:
        # Likely [-1, 1] range; map to [0, 1].
        tensor = (tensor + 1.0) / 2.0
    if tensor.max() > 1.0 or tensor.min() < 0.0:
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def _load_image(path: str) -> torch.Tensor:
    data = load_tensor_or_pil(path)
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
    return _normalize_tensor(tensor)


def _match_channels(tensor: torch.Tensor, channels: int) -> torch.Tensor:
    if tensor.shape[0] == channels:
        return tensor
    if channels == 1 and tensor.shape[0] == 3:
        return tensor.mean(dim=0, keepdim=True)
    if channels == 3 and tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    raise ValueError(f"Cannot match {tensor.shape[0]} channels to {channels}")


def _resize_to(tensor: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[-2:] == size_hw:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _reset_perceptual(metrics: PerceptualMetrics) -> None:
    metrics.fid.reset()
    metrics.kid.reset()
    metrics.lpips.reset()
    metrics.lpips_sum = 0.0
    metrics.sample_count = 0


def _format_metrics(metrics: Dict[str, float]) -> str:
    kid_std = metrics.get("kid_std")
    parts = [
        f"psnr={metrics['psnr']:.4f}",
        f"ssim={metrics['ssim']:.4f}",
        f"lpips={metrics['lpips']:.4f}",
        f"fid={metrics['fid']:.4f}",
        f"kid={metrics['kid']:.6f}",
    ]
    if kid_std is not None:
        parts.append(f"kid_std={kid_std:.6f}")
    return " ".join(parts)


def _write_csv(rows: List[Dict[str, float]], path: str) -> None:
    import csv

    fieldnames = ["real", "pred", "count", "psnr", "ssim", "lpips", "fid", "kid", "kid_std"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_config(path: str | None) -> Dict[str, object]:
    if path is None:
        return {}
    config = load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [value]
    raise ValueError("preds must be a list or a comma-separated string")


def _resolve_path(path: str, base_dir: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(base_dir, expanded))


def _resolve_paths(paths: Sequence[str], base_dir: str) -> List[str]:
    return [_resolve_path(path, base_dir) for path in paths]


def _list_dir_files(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in DEFAULT_EXTS],
        key=lambda p: p.name,
    )


def _build_pairs(real_path: Path, pred_path: Path) -> Tuple[List[Tuple[Path, Path]], str]:
    real_is_dir = real_path.is_dir()
    pred_is_dir = pred_path.is_dir()

    if real_is_dir and not pred_is_dir:
        raise ValueError("real is a directory, pred must also be a directory")

    if real_is_dir and pred_is_dir:
        real_files = {p.name: p for p in _list_dir_files(real_path)}
        pred_files = _list_dir_files(pred_path)
        pairs = [(real_files[p.name], p) for p in pred_files if p.name in real_files]
        return pairs, "dir-dir"

    if (not real_is_dir) and pred_is_dir:
        pred_files = _list_dir_files(pred_path)
        pairs = [(real_path, p) for p in pred_files]
        return pairs, "file-dir"

    return [(real_path, pred_path)], "file-file"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS/FID/KID for generated images vs a real image."
    )
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument("--real", help="Path to the real/reference image.")
    parser.add_argument(
        "--preds",
        nargs="+",
        help="One or more generated image paths.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable resizing preds to match the real image size.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write a CSV report.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    base_dir = os.path.dirname(args.config) if args.config else os.getcwd()

    real_path = args.real or config.get("real")
    preds_list = args.preds if args.preds is not None else _ensure_list(config.get("preds"))
    if not real_path or not preds_list:
        raise ValueError("Both real and preds must be provided via CLI or config.")

    real_path = _resolve_path(str(real_path), base_dir)
    preds_list = _resolve_paths([str(p) for p in preds_list], base_dir)

    resize_cfg = config.get("resize", True)
    if "no_resize" in config:
        resize_cfg = not bool(config.get("no_resize"))
    no_resize = args.no_resize or (not bool(resize_cfg))

    csv_path = args.csv or config.get("csv")
    if csv_path:
        csv_path = _resolve_path(str(csv_path), base_dir)

    device = torch.device(
        args.device
        if args.device is not None
        else (
            str(config.get("device"))
            if config.get("device") is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    )

    real_path_obj = Path(real_path)
    if not real_path_obj.exists():
        raise FileNotFoundError(f"real path not found: {real_path_obj}")

    rows: List[Dict[str, float]] = []

    for pred_path in preds_list:
        pred_path_obj = Path(pred_path)
        if not pred_path_obj.exists():
            raise FileNotFoundError(f"pred path not found: {pred_path_obj}")

        pairs, mode = _build_pairs(real_path_obj, pred_path_obj)
        if not pairs:
            print(f"warning: no matching files for {pred_path_obj}", file=sys.stderr)
            continue

        real_first = _load_image(str(pairs[0][0]))
        real_channels = int(real_first.shape[0])
        real_size = (int(real_first.shape[1]), int(real_first.shape[2]))

        perceptual = PerceptualMetrics.create(device, dataset_size=len(pairs))
        psnr_sum = 0.0
        ssim_sum = 0.0

        with torch.no_grad():
            for real_file, pred_file in pairs:
                real = _load_image(str(real_file))
                pred = _load_image(str(pred_file))
                real = _match_channels(real, real_channels)
                pred = _match_channels(pred, real_channels)
                if not no_resize:
                    real = _resize_to(real, real_size)
                    pred = _resize_to(pred, real_size)
                elif pred.shape[-2:] != real.shape[-2:]:
                    raise ValueError(
                        f"Size mismatch for {pred_file}: {pred.shape[-2:]} vs {real.shape[-2:]}"
                    )

                pred_b = pred.unsqueeze(0).to(device)
                real_b = real.unsqueeze(0).to(device)
                psnr_sum += float(psnr(pred_b, real_b).item())
                ssim_sum += float(ssim(pred_b, real_b).item())
                try:
                    perceptual.update(pred_b, real_b)
                except RuntimeError as exc:
                    print(f"warning: perceptual metrics failed for {pred_file}: {exc}", file=sys.stderr)

            per_metrics = perceptual.compute()

        count = len(pairs)
        row = {
            "real": str(real_path_obj),
            "pred": str(pred_path_obj),
            "count": count,
            "psnr": psnr_sum / max(count, 1),
            "ssim": ssim_sum / max(count, 1),
            "lpips": per_metrics["lpips"],
            "fid": per_metrics["fid"],
            "kid": per_metrics["kid"],
            "kid_std": per_metrics.get("kid_std", float("nan")),
        }
        rows.append(row)
        suffix = f" count={count} mode={mode}" if count > 1 else ""
        print(f"{pred_path_obj} {_format_metrics(row)}{suffix}")

    if csv_path:
        _write_csv(rows, csv_path)
        print(f"wrote csv: {csv_path}")


if __name__ == "__main__":
    main()
