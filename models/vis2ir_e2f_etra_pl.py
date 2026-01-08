from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch.optim import Adam

from flow_matching.path import CondOTProbPath
from models.etra_pl import ETRAPlModule
from models.vis2ir_etrl_pl import Vis2IRETRLPlModule
from utils.flow_sampling import build_solver, sample_ir
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim


class FlowWrapper(torch.nn.Module):
    def __init__(self, unet_model: UNet2DModel):
        super().__init__()
        self.unet = unet_model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(sample=x, timestep=t).sample


class Vis2IRE2FETRALightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.etrl = self._load_etrl(config.get("etrl", {}) or {})
        self.etra = None
        etra_cfg = config.get("etra", {}) or {}
        self.etra_loss_weight = float(etra_cfg.get("loss_weight", 0.0))
        self.etra = self._load_etra(etra_cfg)

        flow_cfg = config.get("flow_model", {}) or {}
        unet_params = flow_cfg.copy()

        if "block_out_channels" not in unet_params:
            base_channels = int(unet_params.pop("base_channels", 32))
            channel_mults = unet_params.pop("channel_mults", [1, 2, 4])
            unet_params["block_out_channels"] = tuple(base_channels * m for m in channel_mults)

        num_levels = len(unet_params["block_out_channels"])
        if "down_block_types" not in unet_params:
            unet_params["down_block_types"] = tuple("DownBlock2D" for _ in range(num_levels))
        if "up_block_types" not in unet_params:
            unet_params["up_block_types"] = tuple("UpBlock2D" for _ in range(num_levels))

        unet_params.update({
            "sample_size": int(config.get("image_size", 64)),
            "in_channels": 1,
            "out_channels": 1,
        })

        self.model = FlowWrapper(UNet2DModel(**unet_params))
        self.path = CondOTProbPath()
        self.solver = build_solver(self.model)

    def _load_etrl(self, etrl_cfg: Dict[str, Any]) -> Vis2IRETRLPlModule:
        etrl_checkpoint = etrl_cfg.get("checkpoint")
        if not etrl_checkpoint:
            raise ValueError("etrl.checkpoint must be provided")

        override_cfg = etrl_cfg.get("config_override")
        if override_cfg:
            etrl = Vis2IRETRLPlModule.load_from_checkpoint(
                etrl_checkpoint,
                config=override_cfg,
                map_location="cpu",
            )
        else:
            etrl = Vis2IRETRLPlModule.load_from_checkpoint(
                etrl_checkpoint,
                map_location="cpu",
            )

        etrl.eval()
        for param in etrl.parameters():
            param.requires_grad = False
        return etrl

    def _load_etra(self, etra_cfg: Dict[str, Any]) -> ETRAPlModule | None:
        etra_enabled = bool(etra_cfg.get("enabled", True))
        if not etra_enabled:
            return None

        etra_checkpoint = etra_cfg.get("checkpoint")
        if etra_checkpoint:
            etra_params = etra_cfg.get("model_params", {}) or {}
            etra = ETRAPlModule.load_from_checkpoint(
                etra_checkpoint,
                map_location="cpu",
                **etra_params,
            )
            etra.eval()
            for param in etra.parameters():
                param.requires_grad = False
            return etra

        if self.etra_loss_weight > 0:
            raise ValueError("etra.loss_weight > 0 but no etra.checkpoint provided")
        return None

    @staticmethod
    def _resolve_sampling_cfg(config: Dict[str, Any], for_etra: bool) -> Dict[str, object]:
        if for_etra:
            return config.get("etra_sampling", config.get("flow_sampling", {})) or {}
        return config.get("flow_sampling", {}) or {}

    def _etrl_forward(self, vis: torch.Tensor) -> torch.Tensor:
        self.etrl.eval()
        with torch.no_grad():
            pred = self.etrl(vis).clamp(0.0, 1.0)
        return pred

    def _etra_guidance_loss(self, pred: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.etra is None:
            return torch.tensor(0.0, device=pred.device), {}
        self.etra.eval()
        decoded = self.etra._decode(self.etra(pred))
        i_hat, tau_low, a_low, r_low = self.etra._reconstruct(decoded)
        loss, loss_dict = self.etra._compute_losses(i_hat, pred, decoded, tau_low, a_low, r_low)
        return loss, loss_dict

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        ir0 = self._etrl_forward(vis)

        t = torch.rand(vis.shape[0], device=self.device)
        path_sample = self.path.sample(x_0=ir0, x_1=ir, t=t)
        pred = self(path_sample.x_t, t)

        flow_loss = F.mse_loss(pred, path_sample.dx_t)
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
        ir0 = self._etrl_forward(vis)
        sampling_cfg = self._resolve_sampling_cfg(self.config, for_etra=False)
        pred_ir = sample_ir(self.solver, cond=None, sampling_cfg=sampling_cfg, x_init=ir0)
        pred_ir = pred_ir.clamp(0.0, 1.0)

        val_psnr = psnr(pred_ir, ir).mean()
        val_ssim = ssim(pred_ir, ir).mean()

        self.log("val/psnr", val_psnr, prog_bar=True)
        self.log("val/ssim", val_ssim, prog_bar=True)
        return {"val_psnr": val_psnr, "val_ssim": val_ssim}

    def configure_optimizers(self):
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
                    "frequency": 1,
                }
            }
        return optimizer
