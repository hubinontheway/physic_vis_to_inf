from __future__ import annotations

import argparse
import math
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from datasets import create_dataset
from models.etra_pl import ETRAPlModule
from utils.config import load_yaml
from utils.vision import load_tensor_or_pil, paired_transform


class IRWrapper(Dataset):
    """Wraps PairedImageDataset to return only IR images as 'image' key."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        # ETRAPlModule expects 'image' key
        return {"image": data["ir"]}


class ImageLogger(Callback):
    """Logs validation images using the model's log_images capability."""
    def __init__(self, val_batch, log_interval=1000):
        super().__init__()
        self.val_batch = val_batch
        self.log_interval = log_interval

    def on_validation_epoch_end(self, trainer, pl_module):
        # We check global step or epoch, but here we can just do it every validation epoch
        # if the validation interval is controlled by trainer.
        
        # Move batch to device
        batch = {k: v.to(pl_module.device) for k, v in self.val_batch.items()}

        # Generate images
        with torch.no_grad():
            images = pl_module.log_images(batch, split="val")

        # Log images to TensorBoard
        if isinstance(trainer.logger, TensorBoardLogger):
            writer = trainer.logger.experiment
            
            # images is a dict of tensors (B, C, H, W)
            # We want to stack them to show: Input | Recon | T | R | A ...
            
            # Select keys to display
            keys = ["input", "recon", "t", "r", "a", "b", "eps", "tau"]
            
            # Helper to normalize -1..1 to 0..1 for display
            def norm(t):
                return (t + 1.0) / 2.0
            
            # Take the first sample from the batch for detailed view, or a few samples
            # Let's grid all samples in the batch (usually 4-8)
            
            # We construct a grid where rows are samples and columns are components
            # But make_grid expects (B, C, H, W). 
            # We can create a list of (B, C, H, W) and concat along width?
            
            vis_list = []
            for k in keys:
                if k in images:
                    img = norm(images[k])
                    if img.shape[1] == 1:
                        img = img.repeat(1, 3, 1, 1)
                    vis_list.append(img)
            
            if vis_list:
                # cat along width (dim 3)
                # grid_row: (B, 3, H, W*len(keys))
                grid_row = torch.cat(vis_list, dim=3)
                
                # Make grid of rows
                grid = make_grid(grid_row, nrow=1)
                
                writer.add_image("val/decomposition", grid, trainer.global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/etra/etra_flat.yml")
    parser.add_argument("--steps", type=int, help="Override max steps")
    args = parser.parse_args()

    config = load_yaml(args.config)
    pl.seed_everything(config.get("seed", 123))

    # --- Config Params ---
    batch_size = int(config.get("batch_size", 8))
    image_size = int(config.get("image_size", 256))
    dataset_name = str(config.get("dataset", "VEDIA"))
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    max_steps = args.steps if args.steps is not None else int(config.get("steps", 100000))
    val_interval = int(config.get("eval_interval", 1000))

    # --- Data ---
    # We use paired_transform but we only care about IR mostly. 
    # However, create_dataset expects paired transform.
    def transform_fn(v, r):
        return paired_transform(v, r, image_size)

    train_dataset_base = create_dataset(
        dataset_name,
        root=dataset_root,
        split=config.get("split", "train"),
        loader=load_tensor_or_pil,
        transform=transform_fn,
        vis_mode="RGB",
        ir_mode="L",
    )
    train_dataset = IRWrapper(train_dataset_base)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    # Validation Set
    val_dataset_base = create_dataset(
        dataset_name,
        root=dataset_root,
        split=config.get("eval_split", "test"),
        loader=load_tensor_or_pil,
        transform=transform_fn,
        vis_mode="RGB",
        ir_mode="L",
    )
    val_dataset = IRWrapper(val_dataset_base)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # --- Model ---
    model_params = config.get("model_params", {})
    # Override learning rate from top level if not in params (or sync them)
    if "base_learning_rate" not in model_params:
        model_params["base_learning_rate"] = float(config.get("lr", 1e-4))
        
    model = ETRAPlModule(**model_params)

    # --- Callbacks & Logger ---
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    logger = TensorBoardLogger("runs", name=config_name)

    checkpoint_callback = ModelCheckpoint(
        filename="step_{step}-loss_{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Image Logger
    try:
        vis_batch = next(iter(val_loader))
        # Keep a fixed subset for visualization (e.g., 4 images)
        fixed_batch = {"image": vis_batch["image"][:4]}
        image_callback = ImageLogger(fixed_batch, log_interval=val_interval)
        callbacks = [checkpoint_callback, lr_monitor, image_callback]
    except StopIteration:
        print("Warning: Validation loader is empty. Skipping ImageLogger.")
        callbacks = [checkpoint_callback, lr_monitor]

    # --- Trainer ---
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

    # Validation scheduling
    num_train_batches = len(train_loader)
    if num_train_batches > 0:
        if val_interval <= num_train_batches:
            val_check_interval = val_interval
            check_val_every_n_epoch = 1
        else:
            val_check_interval = 1.0
            check_val_every_n_epoch = max(1, math.ceil(val_interval / num_train_batches))
        limit_val_batches = 1.0
    else:
        val_check_interval = 1.0
        check_val_every_n_epoch = 1
        limit_val_batches = 0.0

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

    # --- Start Training ---
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
