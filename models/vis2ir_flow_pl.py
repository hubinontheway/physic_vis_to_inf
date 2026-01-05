from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from diffusers import UNet2DModel

# Assuming these are available as per project context
from flow_matching.path import CondOTProbPath
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
        
        # --- Model Initialization (Using diffusers.UNet2DModel) ---
        flow_cfg = config.get("flow_model", {}) or {}
        # Create a copy to avoid modifying the original config
        unet_params = flow_cfg.copy()
        
        cond_mode = str(config.get("phys_align", {}).get("cond_mode", "vis")).lower()
        
        if cond_mode == "vis":
            cond_channels = 3
        elif cond_mode == "phys":
            cond_channels = 2
        else:  # vis_phys
            cond_channels = 5

        # Input channels = IR (1) + Condition (cond_channels)
        in_channels = 1 + cond_channels
        
        # --- Backward Compatibility Logic ---
        # If user provides old-style 'base_channels' and 'channel_mults', convert them to 'block_out_channels'
        if "block_out_channels" not in unet_params:
            base_channels = int(unet_params.pop("base_channels", 32))
            channel_mults = unet_params.pop("channel_mults", [1, 2, 4])
            unet_params["block_out_channels"] = tuple(base_channels * m for m in channel_mults)
            
        # Ensure block types are defined. If not, default to standard ResNet blocks
        num_levels = len(unet_params["block_out_channels"])
        if "down_block_types" not in unet_params:
            unet_params["down_block_types"] = tuple("DownBlock2D" for _ in range(num_levels))
        if "up_block_types" not in unet_params:
            unet_params["up_block_types"] = tuple("UpBlock2D" for _ in range(num_levels))

        # --- Forced/Inferred Parameters ---
        # These are determined by the task structure and cannot be overridden by config
        unet_params.update({
            "sample_size": int(config.get("image_size", 64)),
            "in_channels": in_channels,
            "out_channels": 1,
        })
        
        # Initialize the model with the merged parameters
        self.model = UNet2DModel(**unet_params)
        
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

    def forward(self, x, t, cond):
        # Concatenate condition and input (Standard Image-to-Image pattern)
        # x: (B, 1, H, W), cond: (B, C, H, W) -> model_input: (B, 1+C, H, W)
        model_input = torch.cat([x, cond], dim=1)
        
        # diffusers UNet output is a struct, we need .sample
        return self.model(model_input, t).sample

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

        # Forward pass (using our custom forward which handles concat)
        pred = self(path_sample.x_t, t, cond)
        
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
        cond = self._build_cond(vis, return_phys=False)
        
        # Note: solver.sample calls model(x, t, cond) internally
        # Ensure build_solver wraps the model correctly or the model's forward handles it.
        # Our forward() now handles concat, so it should be compatible if solver passes args correctly.
        
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
        
        steps = int(self.config.get("steps", 10000))
        scheduler = create_lr_scheduler(optimizer, self.config, steps)
        
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        return optimizer
