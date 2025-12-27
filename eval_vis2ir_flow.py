from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from datasets import create_dataset
from models.vis2ir_flow import Vis2IRFlowUNet
from models.vis2phys_align import Vis2PhysAlignUNet
from utils.config import load_yaml
from utils.device import resolve_device
from utils.flow_sampling import build_solver, sample_ir
from utils.metrics import psnr, ssim
from utils.run_artifacts import find_best_checkpoint, find_latest_config, save_eval_metrics
from utils.vision import load_tensor_or_pil, paired_transform, tensor_to_pil


def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    phys_align: Optional[torch.nn.Module] = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    if phys_align is not None:
        if not isinstance(checkpoint, dict) or "phys_align" not in checkpoint:
            raise ValueError("Checkpoint is missing phys_align weights")
        phys_align.load_state_dict(checkpoint["phys_align"], strict=True)


def _build_cond(
    vis: torch.Tensor,
    phys_align: Optional[Vis2PhysAlignUNet],
    cond_mode: str,
    t_cond_norm: bool,
) -> torch.Tensor:
    if cond_mode == "vis":
        return vis
    if phys_align is None:
        raise ValueError("phys_align must be enabled for cond_mode != 'vis'")
    temperature, emissivity = phys_align(vis)
    if t_cond_norm:
        denom = float(phys_align.t_max - phys_align.t_min)
        temperature = (temperature - float(phys_align.t_min)) / denom
        temperature = torch.clamp(temperature, 0.0, 1.0)
    if cond_mode == "phys":
        return torch.cat([temperature, emissivity], dim=1)
    if cond_mode == "vis_phys":
        return torch.cat([vis, temperature, emissivity], dim=1)
    raise ValueError(f"Unsupported cond_mode '{cond_mode}'")


def _resolve_run_dir(run_dir: str | None, config_path: str | None) -> str:
    if run_dir:
        return run_dir
    if config_path:
        config_dir = os.path.dirname(config_path)
        name = os.path.basename(config_path)
        if name.startswith("config_") and name.endswith(".yml") and config_dir:
            return config_dir
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        return os.path.join("runs", config_name)
    raise ValueError("Either --run-dir or --config must be provided")


def run_eval(run_dir: str) -> Dict[str, float]:
    config_path, run_id = find_latest_config(run_dir)
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("train config must be a mapping")

    image_size = int(config["image_size"])
    batch_size = int(config.get("eval_batch_size", config.get("batch_size", 1)))
    eval_split = str(config.get("eval_split", "test"))
    seed = int(config.get("seed", 123))
    torch.manual_seed(seed)

    device = resolve_device(config)

    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))

    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=eval_split,
        loader=load_tensor_or_pil,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="RGB",
        ir_mode="L",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    phys_align_cfg = config.get("phys_align", {}) or {}
    phys_align_enabled = bool(phys_align_cfg.get("enabled", False))
    cond_mode = str(phys_align_cfg.get("cond_mode", "vis")).lower()
    t_cond_norm = bool(phys_align_cfg.get("t_cond_norm", True))
    if not phys_align_enabled and cond_mode != "vis":
        raise ValueError("phys_align.cond_mode requires phys_align.enabled=true")
    if cond_mode == "vis":
        cond_channels = 3
    elif cond_mode == "phys":
        cond_channels = 2
    else:
        cond_channels = 5

    flow_model_cfg = config.get("flow_model", {}) or {}
    base_channels = int(flow_model_cfg.get("base_channels", 32))
    channel_mults = flow_model_cfg.get("channel_mults", [1, 2, 4])
    if not isinstance(channel_mults, (list, tuple)):
        raise ValueError("'flow_model.channel_mults' must be a list of ints")
    channel_mults = tuple(int(v) for v in channel_mults)
    model = Vis2IRFlowUNet(
        in_channels=1,
        cond_channels=cond_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
    ).to(device)
    phys_align = None
    if phys_align_enabled:
        phys_base_channels = int(phys_align_cfg.get("base_channels", 16))
        phys_mults = phys_align_cfg.get("channel_mults", [1, 2, 4])
        if not isinstance(phys_mults, (list, tuple)):
            raise ValueError("'phys_align.channel_mults' must be a list of ints")
        phys_mults = tuple(int(v) for v in phys_mults)
        phys_t_min = float(phys_align_cfg.get("t_min", 1.0))
        phys_t_max = float(phys_align_cfg.get("t_max", 10.0))
        phys_align = Vis2PhysAlignUNet(
            in_channels=3,
            base_channels=phys_base_channels,
            channel_mults=phys_mults,
            t_min=phys_t_min,
            t_max=phys_t_max,
        ).to(device)
    checkpoint_path = find_best_checkpoint(run_dir)
    _load_checkpoint(model, checkpoint_path, device, phys_align=phys_align)
    model.eval()
    if phys_align is not None:
        phys_align.eval()

    solver = build_solver(model)
    sampling_cfg = config.get("flow_sampling", {}) or {}
    sampling_dir = os.path.join(run_dir, "sampling")
    os.makedirs(sampling_dir, exist_ok=True)
    sample_rows = []
    sample_index = 0

    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict) or "vis" not in batch or "ir" not in batch:
                raise ValueError("Expected dataset batch with 'vis' and 'ir' tensors")
            vis = batch["vis"].to(device)
            ir = batch["ir"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            cond = _build_cond(vis, phys_align, cond_mode, t_cond_norm)
            pred_ir = sample_ir(solver, cond, sampling_cfg).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
            batch_psnr = psnr(pred_ir, ir).mean().item()
            batch_ssim = ssim(pred_ir, ir).mean().item()
            batch_size = int(ir.shape[0])
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_samples += batch_size

            per_image = elapsed / max(batch_size, 1)
            peak_mb = None
            if device.type == "cuda":
                peak_bytes = torch.cuda.max_memory_allocated(device)
                peak_mb = peak_bytes / (1024**2)
            pred_cpu = pred_ir.detach().cpu()
            for i in range(batch_size):
                filename = f"sample_{sample_index:06d}.png"
                path = os.path.join(sampling_dir, filename)
                tensor_to_pil(pred_cpu[i], mode="L").save(path)
                sample_rows.append(
                    {
                        "index": sample_index,
                        "filename": filename,
                        "seconds_per_image": round(per_image, 6),
                        "cuda_peak_mb": "" if peak_mb is None else round(peak_mb, 3),
                    }
                )
                sample_index += 1

    if total_samples == 0:
        raise ValueError("Evaluation loader returned no samples")
    metrics = {"psnr": total_psnr / total_samples, "ssim": total_ssim / total_samples}
    save_eval_metrics(run_dir, run_id, metrics, split=eval_split)
    if sample_rows:
        manifest_path = os.path.join(sampling_dir, "sampling_stats.csv")
        with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "filename", "seconds_per_image", "cuda_peak_mb"],
            )
            writer.writeheader()
            writer.writerows(sample_rows)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate flow-matching vis2ir model"
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory containing config_*.yml and checkpoints/",
    )
    parser.add_argument(
        "--config",
        help="Config path; if provided, run_dir defaults to runs/<config_name> or the config's folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir, args.config)
    metrics = run_eval(run_dir)
    print("eval psnr={psnr:.4f} ssim={ssim:.4f}".format(**metrics))


if __name__ == "__main__":
    main()
