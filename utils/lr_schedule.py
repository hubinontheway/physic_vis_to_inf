from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.optim import Optimizer


def _clamp_steps(value: int, total_steps: int) -> int:
    if value < 0:
        raise ValueError("warmup_steps must be >= 0")
    return min(value, max(total_steps, 0))


def _parse_schedule_config(config: Dict[str, object]) -> Optional[Dict[str, object]]:
    schedule = config.get("lr_schedule")
    if schedule is None:
        return None
    if isinstance(schedule, str):
        return {"type": schedule}
    if isinstance(schedule, dict):
        return schedule
    raise ValueError("'lr_schedule' must be a string or mapping")


@dataclass
class WarmupCosineScheduler:
    optimizer: Optimizer
    total_steps: int
    warmup_steps: int
    min_lr: float

    def __post_init__(self) -> None:
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, step: int) -> None:
        if self.total_steps <= 0:
            return
        warmup_steps = _clamp_steps(self.warmup_steps, self.total_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = _compute_warmup_cosine_lr(
                base_lr=base_lr,
                step=step,
                total_steps=self.total_steps,
                warmup_steps=warmup_steps,
                min_lr=self.min_lr,
            )


def _compute_warmup_cosine_lr(
    *,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
) -> float:
    if total_steps <= 0:
        return base_lr
    if step <= 0:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / float(warmup_steps))
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / float(decay_steps), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def create_lr_scheduler(
    optimizer: Optimizer,
    config: Dict[str, object],
    total_steps: int,
) -> Optional[WarmupCosineScheduler]:
    schedule = _parse_schedule_config(config)
    if schedule is None:
        return None
    schedule_type = str(schedule.get("type", "cosine")).lower()
    if schedule_type in {"none", "off", "false"}:
        return None
    if schedule_type != "cosine":
        raise ValueError(f"Unsupported lr_schedule type '{schedule_type}'")

    warmup_steps = int(schedule.get("warmup_steps", 0))
    min_lr = float(schedule.get("min_lr", 0.0))
    if min_lr < 0.0:
        raise ValueError("min_lr must be >= 0")
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=int(total_steps),
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )
