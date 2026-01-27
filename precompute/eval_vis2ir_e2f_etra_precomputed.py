from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from datasets import create_dataset
from datasets.precomputed import PrecomputedIR0Dataset
from precompute.vis2ir_e2f_etra_precomputed import Vis2IRE2FETRAPrecomputed
from utils.config import load_yaml
from utils.device import resolve_device
from utils.flow_sampling import sample_ir
from utils.metrics import PerceptualMetrics, psnr, ssim
from utils.run_artifacts import find_best_checkpoint, find_latest_config, save_eval_metrics
from utils.vision import load_tensor_or_pil, paired_transform, tensor_to_pil


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


def _wrap_precomputed(dataset, config):
    precompute_cfg = config.get("precompute", {}) or {}
    ir0_root = precompute_cfg.get("ir0_root")
    if not ir0_root:
        return dataset
    image_size = int(config.get("image_size", 0)) or None
    use_split_subdir = bool(precompute_cfg.get("use_split_subdir", True))
    return PrecomputedIR0Dataset(
        dataset,
        ir0_root=str(ir0_root),
        image_size=image_size,
        use_split_subdir=use_split_subdir,
    )


def run_eval(run_dir: str) -> Dict[str, float]:
    config_path, run_id = find_latest_config(run_dir)
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("train config must be a mapping")

    if "config" in config and isinstance(config["config"], dict):
        if "dataset" in config["config"]:
            config = config["config"]

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
    dataset = _wrap_precomputed(dataset, config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    checkpoint_path = find_best_checkpoint(run_dir)
    pl_model = Vis2IRE2FETRAPrecomputed.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device,
    )
    pl_model.to(device)
    pl_model.eval()

    sampling_cfg = config.get("flow_sampling", {}) or {}
    sampling_dir = os.path.join(run_dir, "sampling")
    os.makedirs(sampling_dir, exist_ok=True)

    sample_rows = []
    sample_index = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    perceptual_metrics = PerceptualMetrics.create(device, dataset_size=len(dataset))

    with torch.no_grad():
        for batch in loader:
            vis = batch["vis"].to(device)
            ir = batch["ir"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            start_time = time.perf_counter()
            ir0 = pl_model._get_ir0(batch, vis)
            pred_ir = sample_ir(
                pl_model.solver,
                cond=None,
                sampling_cfg=sampling_cfg,
                x_init=ir0,
            ).clamp(0.0, 1.0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time

            batch_psnr = psnr(pred_ir, ir).mean().item()
            batch_ssim = ssim(pred_ir, ir).mean().item()
            b_size = int(ir.shape[0])

            total_psnr += batch_psnr * b_size
            total_ssim += batch_ssim * b_size
            total_samples += b_size
            perceptual_metrics.update(pred_ir, ir)

            per_image = elapsed / max(b_size, 1)
            peak_mb = None
            if device.type == "cuda":
                peak_bytes = torch.cuda.max_memory_allocated(device)
                peak_mb = peak_bytes / (1024**2)

            pred_cpu = pred_ir.cpu()
            path_list = batch.get("ir_path")
            if path_list is None:
                path_list = batch.get("vis_path")
            for i in range(b_size):
                if path_list is not None:
                    filename = os.path.basename(path_list[i])
                else:
                    filename = f"sample_{sample_index:06d}.png"
                tensor_to_pil(pred_cpu[i], mode="L").save(os.path.join(sampling_dir, filename))
                sample_rows.append({
                    "index": sample_index,
                    "filename": filename,
                    "seconds_per_image": round(per_image, 6),
                    "cuda_peak_mb": "" if peak_mb is None else round(peak_mb, 3),
                })
                sample_index += 1

    if total_samples == 0:
        raise ValueError("Evaluation loader returned no samples")

    metrics = {
        "psnr": total_psnr / total_samples,
        "ssim": total_ssim / total_samples,
    }
    metrics.update(perceptual_metrics.compute())

    save_eval_metrics(run_dir, run_id, metrics, split=eval_split)

    if sample_rows:
        manifest_path = os.path.join(sampling_dir, "sampling_stats.csv")
        with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "filename", "seconds_per_image", "cuda_peak_mb"])
            writer.writeheader()
            writer.writerows(sample_rows)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.config)
    metrics = run_eval(run_dir)
    print(
        "eval "
        f"psnr={metrics['psnr']:.4f} "
        f"ssim={metrics['ssim']:.4f} "
        f"lpips={metrics['lpips']:.4f} "
        f"fid={metrics['fid']:.4f} "
        f"kid={metrics['kid']:.6f}"
    )


if __name__ == "__main__":
    main()
