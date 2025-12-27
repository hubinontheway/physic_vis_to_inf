from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader

from datasets import create_dataset
from models.phys_decomp_factory import create_phys_decomp_model
from utils.config import load_yaml
from utils.device import resolve_device
from utils.phys_decomp_data import load_ir_sample, paired_transform
from utils.phys_decomp_metrics import evaluate_model
from utils.run_artifacts import find_best_checkpoint, find_latest_config, save_eval_metrics


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)


def run_eval(run_dir: str) -> Dict[str, float]:
    config_path, run_id = find_latest_config(run_dir)
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("train config must be a mapping")

    image_size = int(config["image_size"])
    batch_size = int(config.get("eval_batch_size", config.get("batch_size", 1)))
    eval_split = str(config.get("eval_split", "test"))
    use_noise = bool(config.get("use_noise", True))
    t_min = float(config.get("t_min", 1.0))
    t_max = float(config.get("t_max", 10.0))

    device = resolve_device(config)

    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))

    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=eval_split,
        loader=load_ir_sample,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="L",
        ir_mode="L",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = create_phys_decomp_model(
        config=config,
        image_size=image_size,
        t_min=t_min,
        t_max=t_max,
        use_noise=use_noise,
        device=device,
    )
    checkpoint_path = find_best_checkpoint(run_dir)
    _load_checkpoint(model, checkpoint_path, device)

    weights = config["loss_weights"]
    metrics = evaluate_model(model, loader, device, weights)
    save_eval_metrics(run_dir, run_id, metrics, split=eval_split)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhysDecomp model")
    parser.add_argument(
        "--run-dir",
        default=os.path.join("runs", "phys_decomp"),
        help="Run directory containing config_*.yml and checkpoints/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_eval(args.run_dir)
    print(
        "eval total={total:.4f} recon={recon:.4f} t_smooth={t_smooth:.4f} "
        "eps_prior={eps_prior:.4f} consistency={consistency:.4f} corr={corr:.4f}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
