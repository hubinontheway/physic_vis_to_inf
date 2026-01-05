from __future__ import annotations

import argparse
import math
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import create_dataset
from models.vis2ir_flow_pl import Vis2IRFlowLightning
from utils.config import load_yaml
from utils.flow_sampling import sample_ir
from utils.vision import load_tensor_or_pil, paired_transform


class ImageLogger(Callback):
    """Logs validation images using the model's sampling capability."""
    def __init__(self, val_samples, num_samples=4, log_interval=1):
        super().__init__()
        # Select a fixed subset for consistent visualization
        self.val_vis = val_samples["vis"][:num_samples]
        self.val_ir = val_samples["ir"][:num_samples]
        self.log_interval = log_interval

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_interval != 0:
            return

        # Move to device
        vis = self.val_vis.to(pl_module.device)
        ir = self.val_ir.to(pl_module.device)

        # Generate
        with torch.no_grad():
            cond = pl_module._build_cond(vis)
            sampling_cfg = pl_module.config.get("flow_sampling", {}) or {}
            pred_ir = sample_ir(pl_module.solver, cond, sampling_cfg).clamp(0.0, 1.0)

        # Log images to TensorBoard
        # Create a grid: Vis | GT IR | Pred IR
        if isinstance(trainer.logger, TensorBoardLogger):
            writer = trainer.logger.experiment
            
            # Helper to normalize for display if needed
            def grid_images(v, gt, pred):
                # v: (B, 3, H, W), gt/pred: (B, 1, H, W)
                gt_3c = gt.repeat(1, 3, 1, 1)
                pred_3c = pred.repeat(1, 3, 1, 1)
                return torch.cat([v, gt_3c, pred_3c], dim=3) # Concatenate horizontally

            grid = grid_images(vis, ir, pred_ir)
            # Log first sample of the grid or make_grid
            from torchvision.utils import make_grid
            grid_image = make_grid(grid, nrow=1)
            
            writer.add_image("val/samples", grid_image, trainer.global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vis2ir_flow_phys_align_teacher.yml")
    parser.add_argument("--steps", type=int, help="Override max steps")
    args = parser.parse_args()

    config = load_yaml(args.config)
    pl.seed_everything(config.get("seed", 123))

    # --- Config Params ---
    batch_size = int(config["batch_size"])
    image_size = int(config["image_size"])
    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    max_steps = args.steps if args.steps is not None else int(config.get("steps", 100000))
    val_interval = int(config.get("eval_interval", 1000)) # Steps, not epochs

    # --- Data ---
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
        pin_memory=True
    )

    # Validation Set
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
        pin_memory=True
    )

    # --- Model ---
    model = Vis2IRFlowLightning(config=config)

    # --- Callbacks & Logger ---
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    logger = TensorBoardLogger("runs", name=config_name)

    checkpoint_callback = ModelCheckpoint(
        filename="step_{step}-psnr_{val/psnr:.2f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Image Logger (take a fixed batch for consistent visualization)
    # We grab a single batch from validation loader
    try:
        vis_batch = next(iter(val_loader))
        image_callback = ImageLogger(vis_batch, num_samples=4, log_interval=1)
        callbacks = [checkpoint_callback, lr_monitor, image_callback]
    except StopIteration:
        print("Warning: Validation loader is empty. Skipping ImageLogger.")
        callbacks = [checkpoint_callback, lr_monitor]

    # --- Trainer ---
    # Accelerate config: check if CUDA is available
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Validation scheduling logic
    # Calculate interval in terms of batches if validation is frequency-based
    num_train_batches = len(train_loader)
    if num_train_batches <= 0:
         # Fallback for empty/dummy datasets
        val_check_interval = 1.0
        check_val_every_n_epoch = 1
        limit_val_batches = 0.0
    else:
        # If val_interval (steps) is smaller than one epoch
        if val_interval <= num_train_batches:
            val_check_interval = val_interval
            check_val_every_n_epoch = 1
        else:
            # If val_interval spans multiple epochs
            val_check_interval = 1.0
            check_val_every_n_epoch = max(1, math.ceil(val_interval / num_train_batches))
        limit_val_batches = 1.0
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == "gpu" else None,
        max_steps=max_steps,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=10,
    )

    # --- Start Training ---
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()