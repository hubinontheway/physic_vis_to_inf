import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from einops import rearrange
from torchmetrics.functional import structural_similarity_index_measure as ssim


class ETRAPlModule(pl.LightningModule):
    """
    ETRA decomposition with physics-inspired reconstruction:
    I_hat = tau * (eps * B(T) + (1-eps) * R) + (1-tau) * A
    """
    def __init__(
        self,
        smp_model="UnetPlusPlus",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet",
        in_channels=1,
        out_channels=5,
        base_learning_rate=1.0e-4,
        weight_decay=0.0,
        lowpass_kernel=16,
        input_range="minus_one_one",
        loss_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_weights"])
        self.learning_rate = base_learning_rate
        self.weight_decay = weight_decay
        self.lowpass_kernel = lowpass_kernel
        self.input_range = input_range

        if encoder_weights == "None":
            encoder_weights = None
        self.net = getattr(smp, smp_model)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )

        # Radiance mapping parameters for B(T)
        self.b_scale = nn.Parameter(torch.tensor(1.0))
        self.b_bias = nn.Parameter(torch.tensor(0.0))

        self.loss_weights = self._build_loss_weights(loss_weights)

    @staticmethod
    def _build_loss_weights(loss_weights):
        default_weights = {
            "rec": 1.0,
            "ssim": 0.2,
            "tv_low": 0.5,
            "tv_eps": 0.1,
            "edge": 0.2,
            "prior": 0.05,
        }
        if loss_weights is None:
            return default_weights
        merged = default_weights.copy()
        merged.update(loss_weights)
        return merged

    def forward(self, x):
        return self.net(x)

    def _get_input(self, batch):
        x = batch["image"] if isinstance(batch, dict) else batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        return x.to(memory_format=torch.contiguous_format).float()

    def _to_01(self, x):
        if self.input_range == "minus_one_one":
            return (x + 1.0) / 2.0
        if self.input_range == "zero_one":
            return x
        raise ValueError(f"Unknown input_range: {self.input_range}")

    @staticmethod
    def _to_minus1_1(x):
        return x * 2.0 - 1.0

    def _decode(self, preds):
        if preds.shape[1] != 5:
            raise ValueError(f"Expected 5 output channels, got {preds.shape[1]}")
        t_raw, e_raw, tau_raw, a_raw, r_raw = torch.chunk(preds, 5, dim=1)
        t = F.softplus(t_raw)
        eps = torch.sigmoid(e_raw)
        tau = torch.sigmoid(tau_raw)
        a = torch.sigmoid(a_raw)
        r = torch.sigmoid(r_raw)
        b_scale = F.softplus(self.b_scale)
        b = torch.sigmoid(b_scale * t + self.b_bias)
        return {"t": t, "eps": eps, "tau": tau, "a": a, "r": r, "b": b}

    def _lowpass(self, x):
        k = int(self.lowpass_kernel)
        if k <= 1:
            return x
        pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
        return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _tv_loss(x):
        dh = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
        dw = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
        return dh + dw

    @staticmethod
    def _edge_loss(t, i):
        t_dx = t[..., 1:] - t[..., :-1]
        i_dx = i[..., 1:] - i[..., :-1]
        t_dy = t[..., 1:, :] - t[..., :-1, :]
        i_dy = i[..., 1:, :] - i[..., :-1, :]
        return (t_dx - i_dx).abs().mean() + (t_dy - i_dy).abs().mean()

    @staticmethod
    def _minmax_norm(x, eps=1e-6):
        min_v = x.amin(dim=(2, 3), keepdim=True)
        max_v = x.amax(dim=(2, 3), keepdim=True)
        return (x - min_v) / (max_v - min_v + eps)

    def _reconstruct(self, decoded):
        tau_low = self._lowpass(decoded["tau"])
        a_low = self._lowpass(decoded["a"])
        r_low = self._lowpass(decoded["r"])
        i_hat = tau_low * (decoded["eps"] * decoded["b"] + (1.0 - decoded["eps"]) * r_low)
        i_hat = i_hat + (1.0 - tau_low) * a_low
        return i_hat, tau_low, a_low, r_low

    def _compute_losses(self, i_hat, i, decoded, tau_low, a_low, r_low):
        weights = self.loss_weights
        loss_rec = F.l1_loss(i_hat, i)
        loss_ssim = 1.0 - ssim(i_hat, i, data_range=1.0)
        loss_tv_low = self._tv_loss(tau_low) + self._tv_loss(a_low) + self._tv_loss(r_low)
        loss_tv_eps = self._tv_loss(decoded["eps"])
        loss_edge = self._edge_loss(decoded["t"], i)
        blur_i = self._lowpass(i)
        loss_prior = F.l1_loss(a_low, blur_i) + F.l1_loss(r_low, blur_i) + torch.mean(torch.abs(1.0 - tau_low))

        loss = (
            weights["rec"] * loss_rec
            + weights["ssim"] * loss_ssim
            + weights["tv_low"] * loss_tv_low
            + weights["tv_eps"] * loss_tv_eps
            + weights["edge"] * loss_edge
            + weights["prior"] * loss_prior
        )

        loss_dict = {
            "loss_rec": loss_rec,
            "loss_ssim": loss_ssim,
            "loss_tv_low": loss_tv_low,
            "loss_tv_eps": loss_tv_eps,
            "loss_edge": loss_edge,
            "loss_prior": loss_prior,
        }
        return loss, loss_dict

    def _shared_step(self, batch, split):
        x = self._get_input(batch)
        i = self._to_01(x)
        preds = self(i)
        decoded = self._decode(preds)
        i_hat, tau_low, a_low, r_low = self._reconstruct(decoded)
        loss, loss_dict = self._compute_losses(i_hat, i, decoded, tau_low, a_low, r_low)

        log_dict = {f"{split}/{k}": v for k, v in loss_dict.items()}
        log_dict[f"{split}/loss"] = loss
        self.log_dict(log_dict, prog_bar=(split == "train"), on_step=True, on_epoch=True, logger=True)
        return loss, i_hat, decoded, tau_low, a_low, r_low, x, i

    def training_step(self, batch, batch_idx):
        loss, *_ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, split="train", **kwargs):
        x = self._get_input(batch)
        i = self._to_01(x)
        preds = self(i)
        decoded = self._decode(preds)
        i_hat, tau_low, a_low, r_low = self._reconstruct(decoded)

        t_vis = self._minmax_norm(decoded["t"])
        images = {
            "input": x,
            "recon": self._to_minus1_1(i_hat),
            "b": self._to_minus1_1(decoded["b"]),
            "t": self._to_minus1_1(t_vis),
            "eps": self._to_minus1_1(decoded["eps"]),
            "tau": self._to_minus1_1(tau_low),
            "a": self._to_minus1_1(a_low),
            "r": self._to_minus1_1(r_low),
        }
        return images
