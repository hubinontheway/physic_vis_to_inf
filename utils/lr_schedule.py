from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_warmup_cosine_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
):
    """
    Returns a lambda function for LambdaLR that implements warmup + cosine decay.
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0) # Clip to [0, 1]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale to range [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return lr_lambda


def create_lr_scheduler(
    optimizer: Optimizer,
    config: Dict[str, object],
    total_steps: int,
) -> Optional[LambdaLR]:
    """
    Creates a standard PyTorch LambdaLR scheduler for warmup cosine decay.
    """
    schedule_cfg = config.get("lr_schedule")
    if schedule_cfg is None:
        return None
        
    if isinstance(schedule_cfg, str):
        schedule_type = schedule_cfg
        schedule_params = {}
    elif isinstance(schedule_cfg, dict):
        schedule_type = schedule_cfg.get("type", "cosine")
        schedule_params = schedule_cfg
    else:
        raise ValueError("'lr_schedule' must be a string or mapping")

    if str(schedule_type).lower() not in {"cosine"}:
        # Return None or raise error depending on strictness. 
        # Original code returned None for "none", "off", "false".
        if str(schedule_type).lower() in {"none", "off", "false"}:
            return None
        raise ValueError(f"Unsupported lr_schedule type '{schedule_type}'")

    warmup_steps = int(schedule_params.get("warmup_steps", 0))
    min_lr = float(schedule_params.get("min_lr", 0.0))
    
    # Calculate min_lr_ratio assuming all param groups have same base LR effectively
    # Or just relative to the initial LR provided in optimizer.
    # Typically min_lr is an absolute value in config, but LambdaLR works on multiplicative factor.
    # To support absolute min_lr correctly with LambdaLR, we need base_lr from optimizer.
    # However, optimizers can have multiple param groups.
    # A safe approximation if we assume base_lr > min_lr:
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    lr_lambda = _get_warmup_cosine_lambda(warmup_steps, total_steps, min_lr_ratio)
    
    return LambdaLR(optimizer, lr_lambda)