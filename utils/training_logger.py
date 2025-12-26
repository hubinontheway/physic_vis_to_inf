from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import torch
from torch import nn

LOSS_COLUMNS = (
    "step",
    "total",
    "recon",
    "t_smooth",
    "eps_prior",
    "consistency",
    "corr",
)


def _format_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _tensor_to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _estimate_model_bytes(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()
    return total


@dataclass(frozen=True)
class TrainingRunLogger:
    run_dir: str
    run_id: str
    config_copy_path: str
    loss_csv_path: str
    metadata_path: str
    loss_columns: tuple[str, ...]

    @classmethod
    def create(
        cls,
        run_dir: str,
        config_path: str,
        loss_columns: tuple[str, ...] | None = None,
    ) -> "TrainingRunLogger":
        os.makedirs(run_dir, exist_ok=True)
        run_id = _format_run_id()
        config_copy_path = os.path.join(run_dir, f"config_{run_id}.yml")
        loss_csv_path = os.path.join(run_dir, f"losses_{run_id}.csv")
        metadata_path = os.path.join(run_dir, f"run_{run_id}.json")
        columns = loss_columns or LOSS_COLUMNS
        if not columns or columns[0] != "step":
            raise ValueError("loss_columns must start with 'step'")
        if os.path.exists(config_path):
            shutil.copyfile(config_path, config_copy_path)
        return cls(
            run_dir=run_dir,
            run_id=run_id,
            config_copy_path=config_copy_path,
            loss_csv_path=loss_csv_path,
            metadata_path=metadata_path,
            loss_columns=tuple(columns),
        )

    def log_model_info(self, model: nn.Module, device: torch.device) -> None:
        param_count = sum(param.numel() for param in model.parameters())
        trainable_count = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        model_bytes = _estimate_model_bytes(model)
        info = {
            "run_id": self.run_id,
            "device": str(device),
            "config_path": self.config_copy_path,
            "param_count": param_count,
            "trainable_param_count": trainable_count,
            "model_size_mb": round(model_bytes / (1024**2), 3),
        }
        if device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            info["cuda_allocated_mb"] = round(allocated / (1024**2), 3)
            info["cuda_reserved_mb"] = round(reserved / (1024**2), 3)
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, sort_keys=True)

    def log_losses(self, step: int, losses: Mapping[str, object]) -> None:
        row = {"step": int(step)}
        for key in self.loss_columns[1:]:
            value = losses.get(key)
            if value is None:
                row[key] = ""
            else:
                row[key] = _tensor_to_float(value)
        file_exists = os.path.exists(self.loss_csv_path)
        with open(self.loss_csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.loss_columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
