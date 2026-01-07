from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import segmentation_models_pytorch as smp

from models.etra_pl import ETRAPlModule
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim


class Vis2IRETRLPlModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        model_params = config.get("model_params", {}) or {}
        smp_model = model_params.get("smp_model", "Unet")
        encoder_name = model_params.get("encoder_name", "resnet34")
        encoder_weights = model_params.get("encoder_weights", "imagenet")
        if encoder_weights == "None":
            encoder_weights = None

        in_channels = int(model_params.get("in_channels", 3))
        out_channels = int(model_params.get("out_channels", 1))

        self.net = getattr(smp, smp_model)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )

        self.output_activation = str(model_params.get("output_activation", "sigmoid")).lower()
        loss_cfg = model_params.get("loss_weights", {}) or {}
        self.loss_weights = {
            "l1": float(loss_cfg.get("l1", 1.0)),
            "ssim": float(loss_cfg.get("ssim", 0.2)),
        }

        self.etra = None
        etra_cfg = config.get("etra", {}) or {}
        self.etra_loss_weight = float(etra_cfg.get("loss_weight", 0.0))
        etra_enabled = bool(etra_cfg.get("enabled", True))
        etra_checkpoint = etra_cfg.get("checkpoint")
        if etra_enabled:
            if etra_checkpoint:
                etra_params = etra_cfg.get("model_params", {}) or {}
                self.etra = ETRAPlModule.load_from_checkpoint(
                    etra_checkpoint,
                    **etra_params,
                    map_location="cpu",
                )
                self.etra.eval()
                for param in self.etra.parameters():
                    param.requires_grad = False
            elif self.etra_loss_weight > 0:
                raise ValueError("etra.loss_weight > 0 but no etra.checkpoint provided")

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        pred = self.net(vis)
        return self._apply_output_activation(pred)

    def _apply_output_activation(self, pred: torch.Tensor) -> torch.Tensor:
        if self.output_activation in {"none", "identity"}:
            return pred
        if self.output_activation == "sigmoid":
            return torch.sigmoid(pred)
        if self.output_activation == "clamp":
            return pred.clamp(0.0, 1.0)
        raise ValueError(f"Unknown output_activation: {self.output_activation}")

    def _etra_guidance_loss(self, pred: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.etra is None:
            return torch.tensor(0.0, device=pred.device), {}
        self.etra.eval()
        decoded = self.etra._decode(self.etra(pred))
        i_hat, tau_low, a_low, r_low = self.etra._reconstruct(decoded)
        loss, loss_dict = self.etra._compute_losses(i_hat, pred, decoded, tau_low, a_low, r_low)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        pred = self(vis).clamp(0.0, 1.0)

        loss_l1 = F.l1_loss(pred, ir)
        loss_ssim = 1.0 - ssim(pred, ir).mean()
        loss = self.loss_weights["l1"] * loss_l1 + self.loss_weights["ssim"] * loss_ssim

        if self.etra is not None and self.etra_loss_weight > 0:
            etra_loss, etra_logs = self._etra_guidance_loss(pred)
            loss = loss + self.etra_loss_weight * etra_loss
            self.log("train/etra_loss", etra_loss, on_step=True, on_epoch=True)
            for key, value in etra_logs.items():
                self.log(f"train/etra_{key}", value, on_step=True, on_epoch=True)

        self.log("train/l1", loss_l1, on_step=True, on_epoch=True)
        self.log("train/ssim_loss", loss_ssim, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, ir = batch["vis"], batch["ir"]
        pred = self(vis).clamp(0.0, 1.0)

        loss_l1 = F.l1_loss(pred, ir)
        loss_ssim = 1.0 - ssim(pred, ir).mean()
        loss = self.loss_weights["l1"] * loss_l1 + self.loss_weights["ssim"] * loss_ssim

        if self.etra is not None and self.etra_loss_weight > 0:
            etra_loss, _ = self._etra_guidance_loss(pred)
            loss = loss + self.etra_loss_weight * etra_loss

        val_psnr = psnr(pred, ir).mean()
        val_ssim = ssim(pred, ir).mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_psnr": val_psnr, "val_ssim": val_ssim}

    def configure_optimizers(self):
        model_params = self.config.get("model_params", {}) or {}
        lr = float(model_params.get("lr", self.config.get("lr", 1e-4)))
        weight_decay = float(model_params.get("weight_decay", self.config.get("weight_decay", 0.0)))
        optimizer = AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        steps = int(self.config.get("steps", 10000))
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
