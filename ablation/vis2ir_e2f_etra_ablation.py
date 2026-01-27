from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.vis2ir_e2f_etra_pl import Vis2IRE2FETRALightning
from utils.metrics import psnr, ssim


class Vis2IRE2FETRAAblation(Vis2IRE2FETRALightning):
    def __init__(self, config: Dict[str, Any]):
        self.ablation = config.get("ablation", {}) or {}
        self.precompute = config.get("precompute", {}) or {}
        self.precompute_use_only = bool(self.precompute.get("use_only", False))
        self.etrl_mode = str(self.ablation.get("etrl_mode", "default")).lower()
        if self.etrl_mode in {"etrl", "baseline"}:
            self.etrl_mode = "default"
        self.etrl_trainable = bool(self.ablation.get("etrl_trainable", False))
        self.etrl_noise_std = float(self.ablation.get("etrl_noise_std", 0.1))
        self.etrl_noise_mean = float(self.ablation.get("etrl_noise_mean", 0.5))
        self.etrl_constant = float(self.ablation.get("etrl_constant", 0.0))
        self.is_oracle = self.etrl_mode == "gt"
        if self.precompute_use_only:
            self.etrl_trainable = False
        if self.etrl_mode != "default":
            self.etrl_trainable = False
        super().__init__(config)

    def _load_etrl(self, etrl_cfg: Dict[str, Any]):
        if self.etrl_mode != "default":
            return None
        if self.precompute_use_only:
            return None
        etrl = super()._load_etrl(etrl_cfg)
        if self.etrl_trainable:
            etrl.train()
            for param in etrl.parameters():
                param.requires_grad = True
        return etrl

    def _etrl_forward(self, vis: torch.Tensor) -> torch.Tensor:
        if self.etrl is None:
            raise ValueError("ETRL is not available for etrl_mode=default")
        if self.etrl_trainable:
            pred = self.etrl(vis)
            return pred.clamp(0.0, 1.0)
        self.etrl.eval()
        with torch.no_grad():
            pred = self.etrl(vis).clamp(0.0, 1.0)
        return pred

    def _get_ir0(self, vis: torch.Tensor, ir: torch.Tensor, batch=None) -> torch.Tensor:
        if self.etrl_mode == "gt":
            return ir
        if self.etrl_mode == "zero":
            return torch.zeros_like(ir)
        if self.etrl_mode == "noise":
            noise = torch.randn_like(ir) * self.etrl_noise_std + self.etrl_noise_mean
            return noise.clamp(0.0, 1.0)
        if self.etrl_mode == "constant":
            return torch.full_like(ir, self.etrl_constant).clamp(0.0, 1.0)
        if batch is not None and "ir0" in batch:
            ir0 = batch["ir0"]
            if isinstance(ir0, torch.Tensor):
                return ir0.to(vis.device)
        if self.precompute_use_only:
            raise ValueError("precompute.use_only=true but batch has no ir0")
        return self._etrl_forward(vis)

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        ir0 = self._get_ir0(vis, ir, batch=batch)

        self.log("ablation/is_oracle", float(self.is_oracle), on_step=False, on_epoch=True)

        t = torch.rand(vis.shape[0], device=self.device)
        path_sample = self.path.sample(x_0=ir0, x_1=ir, t=t)
        pred = self(path_sample.x_t, t)

        flow_loss = F.mse_loss(pred, path_sample.dx_t)
        loss = flow_loss
        self.log("train/flow_loss", flow_loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.etra is not None and self.etra_loss_weight > 0:
            pred_ir = self._etra_guidance_pred_ir(ir0)
            etra_loss, etra_logs = self._etra_guidance_loss(pred_ir)
            loss = loss + self.etra_loss_weight * etra_loss
            self.log("train/etra_loss", etra_loss, on_step=True, on_epoch=True)
            for key, value in etra_logs.items():
                self.log(f"train/etra_{key}", value, on_step=True, on_epoch=True)

        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        ir0 = self._get_ir0(vis, ir, batch=batch)
        sampling_cfg = self._resolve_sampling_cfg(self.config, for_etra=False)
        pred_ir = sample_ir(self.solver, cond=None, sampling_cfg=sampling_cfg, x_init=ir0)
        pred_ir = pred_ir.clamp(0.0, 1.0)

        val_psnr = psnr(pred_ir, ir).mean()
        val_ssim = ssim(pred_ir, ir).mean()

        self.log("val/psnr", val_psnr, prog_bar=True)
        self.log("val/ssim", val_ssim, prog_bar=True)
        return {"val_psnr": val_psnr, "val_ssim": val_ssim}

    def configure_optimizers(self):
        lr = float(self.config.get("lr", 1e-4))
        param_groups = [{"params": list(self.model.parameters()), "lr": lr}]
        if self.etrl_trainable:
            if self.etrl is None:
                raise ValueError("etrl_trainable requires etrl_mode=default")
            etrl_lr = float(self.ablation.get("etrl_lr", lr))
            param_groups.append({"params": list(self.etrl.parameters()), "lr": etrl_lr})
        optimizer = Adam(param_groups, lr=lr)

        steps = int(self.config.get("steps", 10000))
        from utils.lr_schedule import create_lr_scheduler
        scheduler = create_lr_scheduler(optimizer, self.config, steps)
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return optimizer
