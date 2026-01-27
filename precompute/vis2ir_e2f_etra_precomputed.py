from __future__ import annotations

from typing import Any, Dict

import torch

from models.vis2ir_e2f_etra_pl import Vis2IRE2FETRALightning
from utils.flow_sampling import sample_ir
from utils.metrics import psnr, ssim


class Vis2IRE2FETRAPrecomputed(Vis2IRE2FETRALightning):
    def __init__(self, config: Dict[str, Any]):
        self.precompute = config.get("precompute", {}) or {}
        self.use_only = bool(self.precompute.get("use_only", False))
        super().__init__(config)

    def _load_etrl(self, etrl_cfg: Dict[str, Any]):
        if self.use_only:
            return None
        return super()._load_etrl(etrl_cfg)

    def _get_ir0(self, batch, vis: torch.Tensor) -> torch.Tensor:
        if "ir0" in batch:
            ir0 = batch["ir0"]
            if isinstance(ir0, torch.Tensor):
                return ir0.to(vis.device)
        if self.etrl is None:
            raise ValueError("precompute.use_only=true but batch has no ir0")
        return self._etrl_forward(vis)

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        ir0 = self._get_ir0(batch, vis)

        t = torch.rand(vis.shape[0], device=self.device)
        path_sample = self.path.sample(x_0=ir0, x_1=ir, t=t)
        pred = self(path_sample.x_t, t)

        flow_loss = torch.nn.functional.mse_loss(pred, path_sample.dx_t)
        loss = flow_loss
        self.log("train/flow_loss", flow_loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.etra is not None and self.etra_loss_weight > 0:
            sampling_cfg = self._resolve_sampling_cfg(self.config, for_etra=True)
            pred_ir = sample_ir(self.solver, cond=None, sampling_cfg=sampling_cfg, x_init=ir0)
            pred_ir = pred_ir.clamp(0.0, 1.0)
            etra_loss, etra_logs = self._etra_guidance_loss(pred_ir)
            loss = loss + self.etra_loss_weight * etra_loss
            self.log("train/etra_loss", etra_loss, on_step=True, on_epoch=True)
            for key, value in etra_logs.items():
                self.log(f"train/etra_{key}", value, on_step=True, on_epoch=True)

        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        ir0 = self._get_ir0(batch, vis)
        sampling_cfg = self._resolve_sampling_cfg(self.config, for_etra=False)
        pred_ir = sample_ir(self.solver, cond=None, sampling_cfg=sampling_cfg, x_init=ir0)
        pred_ir = pred_ir.clamp(0.0, 1.0)

        val_psnr = psnr(pred_ir, ir).mean()
        val_ssim = ssim(pred_ir, ir).mean()

        self.log("val/psnr", val_psnr, prog_bar=True)
        self.log("val/ssim", val_ssim, prog_bar=True)
        return {"val_psnr": val_psnr, "val_ssim": val_ssim}
