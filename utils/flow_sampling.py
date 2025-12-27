from __future__ import annotations

from typing import Dict

import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


class ConditionalVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        cond = extras.get("cond")
        if cond is None:
            raise ValueError("cond must be provided for conditional sampling")
        return self.model(x=x, t=t, cond=cond)


def build_solver(model: torch.nn.Module) -> ODESolver:
    return ODESolver(ConditionalVelocityWrapper(model))


def sample_ir(
    solver: ODESolver,
    cond: torch.Tensor,
    sampling_cfg: Dict[str, object],
) -> torch.Tensor:
    steps = int(sampling_cfg.get("steps", 50))
    method = str(sampling_cfg.get("method", "euler"))
    step_size = float(sampling_cfg.get("step_size", 1.0 / max(steps, 1)))
    atol = float(sampling_cfg.get("atol", 1e-5))
    rtol = float(sampling_cfg.get("rtol", 1e-5))
    time_grid = torch.tensor([0.0, 1.0], device=cond.device)
    x_init = torch.randn(
        cond.shape[0], 1, cond.shape[-2], cond.shape[-1], device=cond.device
    )
    return solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        atol=atol,
        rtol=rtol,
        time_grid=time_grid,
        cond=cond,
    )
