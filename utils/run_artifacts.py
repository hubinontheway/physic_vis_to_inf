from __future__ import annotations

import glob
import json
import os
from typing import Dict, Tuple


def _latest_by_mtime(paths: list[str]) -> str:
    if not paths:
        raise FileNotFoundError("No matching files found")
    return max(paths, key=lambda p: os.path.getmtime(p))


def _resolve_subdir(run_dir: str) -> str:
    """If run_dir contains version_*, return the latest one. Otherwise return run_dir."""
    # If run_dir itself has artifacts, assume it's the right one
    if os.path.exists(os.path.join(run_dir, "checkpoints")) or \
       glob.glob(os.path.join(run_dir, "config_*.yml")) or \
       os.path.exists(os.path.join(run_dir, "hparams.yaml")):
        return run_dir

    # Check for versions
    versions = glob.glob(os.path.join(run_dir, "version_*"))
    if versions:
        # Sort by version number (assuming "version_N") or mtime
        # version_10 > version_2, so lexical sort on full string is risky.
        # Let's use mtime as it's reliable for "latest run".
        return _latest_by_mtime(versions)
    
    return run_dir

def find_latest_config(run_dir: str) -> Tuple[str, str]:
    target_dir = _resolve_subdir(run_dir)
    
    # Priority 1: config_*.yml
    pattern = os.path.join(target_dir, "config_*.yml")
    configs = glob.glob(pattern)
    
    if configs:
        config_path = _latest_by_mtime(configs)
        run_id = _run_id_from_config(config_path)
        return config_path, run_id
    
    # Priority 2: hparams.yaml
    hparams = os.path.join(target_dir, "hparams.yaml")
    if os.path.exists(hparams):
        # Use parent dir name or version name as run_id
        run_id = os.path.basename(target_dir)
        return hparams, run_id

    raise FileNotFoundError(f"No config_*.yml or hparams.yaml found in {run_dir} (resolved to {target_dir})")

def _run_id_from_config(config_path: str) -> str:
    name = os.path.basename(config_path)
    if name == "hparams.yaml":
        return "hparams"
    if name.startswith("config_") and name.endswith(".yml"):
        return name[len("config_") : -len(".yml")]
    return "unknown"

def find_best_checkpoint(run_dir: str) -> str:
    target_dir = _resolve_subdir(run_dir)
    checkpoint_dir = os.path.join(target_dir, "checkpoints")
    
    # Check for best.pt or best.ckpt
    best_pt = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_pt):
        return best_pt
    best_ckpt = os.path.join(checkpoint_dir, "best.ckpt")
    if os.path.exists(best_ckpt):
        return best_ckpt

    # Candidates: step_*.pt or step_*.ckpt or epoch_*.ckpt or just *.ckpt
    candidates = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    candidates.extend(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    
    if not candidates:
         raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    return _latest_by_mtime(candidates)

def save_eval_metrics(
    run_dir: str,
    run_id: str,
    metrics: Dict[str, float],
    split: str,
) -> str:
    # Save in the resolved directory if possible, or just the provided run_dir
    # To avoid splitting artifacts across base/version dirs, let's try to resolve
    target_dir = _resolve_subdir(run_dir)
    
    os.makedirs(target_dir, exist_ok=True)
    payload = {"split": split, "metrics": metrics}
    path = os.path.join(target_dir, f"eval_{run_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path