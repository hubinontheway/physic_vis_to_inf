from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from diffusers import UNet2DModel

# Assuming these are available as per project context
from flow_matching.path import CondOTProbPath
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim
from utils.flow_sampling import build_solver, sample_ir


class ConcatWrapper(torch.nn.Module):
    def __init__(self, unet_model: UNet2DModel):
        super().__init__()
        self.unet = unet_model
        
    def forward(self, x, t, cond=None):
        # Allow calling with or without kwargs from flow_matching
        # x: (B, 1, H, W), cond: (B, 3, H, W)
        if cond is None:
            raise ValueError("Condition 'cond' is required")
            
        model_input = torch.cat([x, cond], dim=1)
        # Explicitly use keyword arguments for safety
        return self.unet(sample=model_input, timestep=t).sample


class Vis2IRFlowLightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # --- Model Initialization (Using diffusers.UNet2DModel) ---
        flow_cfg = config.get("flow_model", {}) or {}
        # Create a copy to avoid modifying the original config
        unet_params = flow_cfg.copy()
        
        # Hardcode channels for 'vis' (RGB) -> IR (L) task
        # Input channels = IR (1) + Vis Condition (3)
        cond_channels = 3
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
        
        # Initialize the diffusers model
        unet = UNet2DModel(**unet_params)
        
        # Wrap it to handle concatenation
        # This wrapper will be the main 'model' attribute
        self.model = ConcatWrapper(unet)
        
        # Flow matching path
        self.path = CondOTProbPath()

    def setup(self, stage: str = None):
        # Build solver for sampling/validation
        # Now we can safely pass self.model (ConcatWrapper) because it's a child module,
        # not the LightningModule itself.
        self.solver = build_solver(self.model)

    def _build_cond(self, vis: torch.Tensor):
        # Simple passthrough for vis-only conditioning
        return vis

    def forward(self, x, t, cond):
        # Delegate to the wrapper
        return self.model(x, t, cond=cond)

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        
        # Flow matching setup
        t = torch.rand(vis.shape[0], device=self.device)
        x0 = torch.randn_like(ir)
        path_sample = self.path.sample(x_0=x0, x_1=ir, t=t)
        
        # Condition building
        cond = self._build_cond(vis)

        # Forward pass (using our custom forward which handles concat)
        pred = self(path_sample.x_t, t, cond)
        
        # Flow loss
        loss = F.mse_loss(pred, path_sample.dx_t)
        self.log("train/flow_loss", loss, prog_bar=True)
        self.log("train/total_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        sampling_cfg = self.config.get("flow_sampling", {}) or {}
        
        # For validation, we generate samples
        cond = self._build_cond(vis)
        
        # Note: solver.sample calls model(x, t, cond) internally
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