from __future__ import annotations

import glob
import json
import os
from typing import Dict, Tuple


def _latest_by_mtime(paths: list[str]) -> str:
    if not paths:
        raise FileNotFoundError("No matching files found")
    return max(paths, key=lambda p: os.path.getmtime(p))


def find_latest_config(run_dir: str) -> Tuple[str, str]:
    pattern = os.path.join(run_dir, "config_*.yml")
    configs = glob.glob(pattern)
    config_path = _latest_by_mtime(configs)
    run_id = _run_id_from_config(config_path)
    return config_path, run_id


def _run_id_from_config(config_path: str) -> str:
    name = os.path.basename(config_path)
    if not name.startswith("config_") or not name.endswith(".yml"):
        raise ValueError(f"Unexpected config filename '{name}'")
    return name[len("config_") : -len(".yml")]


def find_best_checkpoint(run_dir: str) -> str:
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        return best_path
    candidates = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    return _latest_by_mtime(candidates)


def save_eval_metrics(
    run_dir: str,
    run_id: str,
    metrics: Dict[str, float],
    split: str,
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    payload = {"split": split, "metrics": metrics}
    path = os.path.join(run_dir, f"eval_{run_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
