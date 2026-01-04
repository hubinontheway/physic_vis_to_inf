from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Assuming these are available as per project context
from flow_matching.path import CondOTProbPath
from models.vis2ir_flow import Vis2IRFlowUNet
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim
from utils.flow_sampling import build_solver, sample_ir

class Vis2IRFlowLightning(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # --- Model Initialization ---
        flow_cfg = config.get("flow_model", {}) or {}
        cond_channels = 3
        self.model = Vis2IRFlowUNet(
            in_channels=1,
            cond_channels=cond_channels,
            base_channels=int(flow_cfg.get("base_channels", 32)),
            channel_mults=tuple(int(v) for v in flow_cfg.get("channel_mults", [1, 2, 4])),
        )
        
        # Flow matching path
        self.path = CondOTProbPath()

    def setup(self, stage: str = None):
        # Build solver for sampling/validation
        self.solver = build_solver(self.model)

    def _build_cond(self, vis: torch.Tensor) -> torch.Tensor:
        return vis

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        
        # Flow matching setup
        t = torch.rand(vis.shape[0], device=self.device)
        x0 = torch.randn_like(ir)
        path_sample = self.path.sample(x_0=x0, x_1=ir, t=t)
        
        # Condition building
        cond = self._build_cond(vis)

        # Forward pass
        pred = self.model(path_sample.x_t, t, cond=cond)
        
        # Flow loss
        loss = F.mse_loss(pred, path_sample.dx_t)
        self.log("train/flow_loss", loss, prog_bar=True)

        self.log("train/total_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        sampling_cfg = self.config.get("flow_sampling", {}) or {}
        
        # For validation, we generate samples
        # Note: Sampling can be slow, might want to limit this
        cond = self._build_cond(vis)
        pred_ir = sample_ir(self.solver, cond, sampling_cfg).clamp(0.0, 1.0)
        
        # Metrics
        val_psnr = psnr(pred_ir, ir).mean()
        val_ssim = ssim(pred_ir, ir).mean()
        
        self.log("val/psnr", val_psnr, prog_bar=True)
        self.log("val/ssim", val_ssim, prog_bar=True)
        
        return {"val_psnr": val_psnr, "val_ssim": val_ssim}

    def configure_optimizers(self):
        # Collect parameters
        params = list(self.model.parameters())
            
        lr = float(self.config.get("lr", 1e-4))
        optimizer = Adam(params, lr=lr)
        
        # Use existing scheduler logic or simple one
        # Because create_lr_scheduler implies a custom loop, we might need to adapt it.
        # For now, let's assume it returns a standard torch scheduler or None
        steps = int(self.config.get("steps", 10000))
        scheduler = create_lr_scheduler(optimizer, self.config, steps)
        
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # Assuming step-based scheduler from original code
                    "frequency": 1
                }
            }
        return optimizer
