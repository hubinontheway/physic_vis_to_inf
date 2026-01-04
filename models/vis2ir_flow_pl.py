from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Assuming these are available as per project context
from flow_matching.path import CondOTProbPath
from models.vis2ir_flow import Vis2IRFlowUNet
from models.vis2phys_align import Vis2PhysAlignUNet
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim
from utils.flow_sampling import build_solver, sample_ir

class Vis2IRFlowLightning(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        phys_align_model: Optional[Vis2PhysAlignUNet] = None,
        teacher_model: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["phys_align_model", "teacher_model"])
        self.config = config
        
        # --- Model Initialization ---
        flow_cfg = config.get("flow_model", {}) or {}
        cond_mode = str(config.get("phys_align", {}).get("cond_mode", "vis")).lower()
        
        if cond_mode == "vis":
            cond_channels = 3
        elif cond_mode == "phys":
            cond_channels = 2
        else:  # vis_phys
            cond_channels = 5

        self.model = Vis2IRFlowUNet(
            in_channels=1,
            cond_channels=cond_channels,
            base_channels=int(flow_cfg.get("base_channels", 32)),
            channel_mults=tuple(int(v) for v in flow_cfg.get("channel_mults", [1, 2, 4])),
        )
        
        # Optional components
        self.phys_align = phys_align_model
        self.phys_teacher = teacher_model
        
        # Flow matching path
        self.path = CondOTProbPath()
        
        # Configuration extraction for easy access
        self.phys_align_cfg = config.get("phys_align", {}) or {}
        self.cond_mode = cond_mode
        self.t_cond_norm = bool(self.phys_align_cfg.get("t_cond_norm", True))
        
        # Teacher config
        teacher_cfg = self.phys_align_cfg.get("teacher", {}) or {}
        self.teacher_loss_weight = float(teacher_cfg.get("loss_weight", 1.0))
        self.teacher_t_weight = float(teacher_cfg.get("t_weight", 1.0))
        self.teacher_eps_weight = float(teacher_cfg.get("eps_weight", 1.0))

    def setup(self, stage: str = None):
        # Build solver for sampling/validation
        self.solver = build_solver(self.model)

    def _build_cond(self, vis: torch.Tensor, return_phys: bool = False):
        if self.cond_mode == "vis":
            return (vis, None, None) if return_phys else vis

        if self.phys_align is None:
             # Should be handled by config validation, but safe check
            raise ValueError("phys_align model required for non-vis cond_mode")

        temperature, emissivity = self.phys_align(vis)
        
        # Normalize temperature if configured
        temperature_cond = temperature
        if self.t_cond_norm:
            # Access attributes from phys_align model assuming they exist
            t_min = getattr(self.phys_align, 't_min', 1.0)
            t_max = getattr(self.phys_align, 't_max', 10.0)
            denom = float(t_max - t_min)
            temperature_cond = (temperature - float(t_min)) / denom
            temperature_cond = torch.clamp(temperature_cond, 0.0, 1.0)

        if self.cond_mode == "phys":
            cond = torch.cat([temperature_cond, emissivity], dim=1)
        elif self.cond_mode == "vis_phys":
            cond = torch.cat([vis, temperature_cond, emissivity], dim=1)
        else:
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")
            
        if return_phys:
            return cond, temperature, emissivity
        return cond

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        
        # Flow matching setup
        t = torch.rand(vis.shape[0], device=self.device)
        x0 = torch.randn_like(ir)
        path_sample = self.path.sample(x_0=x0, x_1=ir, t=t)
        
        # Condition building
        cond_result = self._build_cond(
            vis, 
            return_phys=(self.phys_teacher is not None)
        )
        
        if self.phys_teacher is not None:
            cond, phys_t_pred, phys_eps_pred = cond_result
        else:
            cond = cond_result
            phys_t_pred, phys_eps_pred = None, None

        # Forward pass
        pred = self.model(path_sample.x_t, t, cond=cond)
        
        # Flow loss
        loss = F.mse_loss(pred, path_sample.dx_t)
        self.log("train/flow_loss", loss, prog_bar=True)

        # Teacher loss (Physical Alignment)
        if self.phys_teacher is not None:
            with torch.no_grad():
                teacher_t, teacher_eps, _, _ = self.phys_teacher(ir)
            
            t_loss = F.l1_loss(phys_t_pred, teacher_t)
            eps_loss = F.l1_loss(phys_eps_pred, teacher_eps)
            
            phys_loss = self.teacher_loss_weight * (
                self.teacher_t_weight * t_loss + self.teacher_eps_weight * eps_loss
            )
            loss = loss + phys_loss
            
            self.log("train/phys_loss", phys_loss)
            self.log("train/t_loss", t_loss)
            self.log("train/eps_loss", eps_loss)

        self.log("train/total_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        sampling_cfg = self.config.get("flow_sampling", {}) or {}
        
        # For validation, we generate samples
        # Note: Sampling can be slow, might want to limit this
        cond = self._build_cond(vis, return_phys=False)
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
        if self.phys_align is not None:
            params += list(self.phys_align.parameters())
            
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
