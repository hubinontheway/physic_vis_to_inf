from __future__ import annotations

import argparse
import math
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import create_dataset
from models.vis2ir_e2f_etra_pl import Vis2IRE2FETRALightning
from utils.config import load_yaml
from utils.flow_sampling import sample_ir
from utils.vision import load_tensor_or_pil, paired_transform


class ImageLogger(Callback):
    """Logs validation samples: IR0 | GT IR | Pred IR."""
    def __init__(self, val_samples, num_samples=4, log_interval=1):
        super().__init__()
        self.val_vis = val_samples["vis"][:num_samples]
        self.val_ir = val_samples["ir"][:num_samples]
        self.log_interval = log_interval

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_interval != 0:
            return

        vis = self.val_vis.to(pl_module.device)
        ir = self.val_ir.to(pl_module.device)

        with torch.no_grad():
            ir0 = pl_module._etrl_forward(vis)
            sampling_cfg = pl_module.config.get("flow_sampling", {}) or {}
            pred_ir = sample_ir(
                pl_module.solver,
                cond=None,
                sampling_cfg=sampling_cfg,
                x_init=ir0,
            ).clamp(0.0, 1.0)

        if isinstance(trainer.logger, TensorBoardLogger):
            writer = trainer.logger.experiment

            def _grid_images(seed, gt, pred):
                seed_3c = seed.repeat(1, 3, 1, 1)
                gt_3c = gt.repeat(1, 3, 1, 1)
                pred_3c = pred.repeat(1, 3, 1, 1)
                return torch.cat([seed_3c, gt_3c, pred_3c], dim=3)

            grid = _grid_images(ir0, ir, pred_ir)
            from torchvision.utils import make_grid
            grid_image = make_grid(grid, nrow=1)
            writer.add_image("val/samples", grid_image, trainer.global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow/vis2ir_e2f_etra.yml")
    parser.add_argument("--steps", type=int, help="Override max steps")
    args = parser.parse_args()

    config = load_yaml(args.config)
    pl.seed_everything(config.get("seed", 123))

    batch_size = int(config["batch_size"])
    image_size = int(config["image_size"])
    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    max_steps = args.steps if args.steps is not None else int(config.get("steps", 100000))
    val_interval = int(config.get("eval_interval", 1000))
    vis_interval = int(config.get("vis_interval", 1))

    train_dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=config.get("split", "train"),
        loader=load_tensor_or_pil,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="RGB",
        ir_mode="L",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=config.get("eval_split", "test"),
        loader=load_tensor_or_pil,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="RGB",
        ir_mode="L",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = Vis2IRE2FETRALightning(config=config)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    logger = TensorBoardLogger("runs", name=config_name)

    best_metric = str(config.get("best_metric", "psnr"))
    best_metric_mode = str(config.get("best_metric_mode", "max"))
    max_checkpoints = int(config.get("max_checkpoints", 5))

    checkpoint_callback = ModelCheckpoint(
        filename=f"step_{{step}}-{best_metric}_{{val/{best_metric}:.2f}}",
        monitor=f"val/{best_metric}",
        mode=best_metric_mode,
        save_top_k=max_checkpoints,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    try:
        vis_batch = next(iter(val_loader))
        image_callback = ImageLogger(vis_batch, num_samples=4, log_interval=vis_interval)
        callbacks = [checkpoint_callback, lr_monitor, image_callback]
    except StopIteration:
        print("Warning: Validation loader is empty. Skipping ImageLogger.")
        callbacks = [checkpoint_callback, lr_monitor]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device_cfg = str(config.get("device", "0")).lower()
    if accelerator == "gpu":
        if ":" in device_cfg:
            devices = [int(device_cfg.split(":")[-1])]
        elif device_cfg.isdigit():
            devices = [int(device_cfg)]
        else:
            devices = 1
    else:
        devices = "auto"

    num_train_batches = len(train_loader)
    if num_train_batches <= 0:
        val_check_interval = 1.0
        check_val_every_n_epoch = 1
        limit_val_batches = 0.0
    else:
        if val_interval <= num_train_batches:
            val_check_interval = val_interval
            check_val_every_n_epoch = 1
        else:
            val_check_interval = 1.0
            check_val_every_n_epoch = max(1, math.ceil(val_interval / num_train_batches))
        limit_val_batches = 1.0

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=10,
    )

    log_dir = trainer.log_dir or trainer.logger.log_dir
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        import time
        import yaml
        timestamp = int(time.time())
        config_save_path = os.path.join(log_dir, f"config_{timestamp}.yml")
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        print(f"Config saved to {config_save_path}")

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
