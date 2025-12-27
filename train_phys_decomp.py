from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterator, Tuple

import torch
from torch.utils.data import DataLoader

from datasets import create_dataset
from models.phys_decomp_factory import create_phys_decomp_model
from utils.config import load_yaml
from utils.device import resolve_device
from utils.phys_decomp_data import load_ir_sample, paired_transform
from utils.phys_decomp_metrics import compute_losses, evaluate_model
from utils.lr_schedule import create_lr_scheduler
from utils.training_logger import TrainingRunLogger

def _sample_geom_params(tensor: torch.Tensor, size: int) -> Tuple[int, int, bool, bool]:
    """
    Sample geometric augmentation parameters for an image tensor.
    
    This function randomly samples parameters for cropping and flipping operations
    that will be applied to an image. This is used for data augmentation during
    training to improve model robustness.
    
    Args:
        tensor (torch.Tensor): Input image tensor of shape (C, H, W)
        size (int): Target crop size
        
    Returns:
        Tuple containing:
        - top (int): Top coordinate for cropping
        - left (int): Left coordinate for cropping
        - flip_h (bool): Whether to flip horizontally
        - flip_v (bool): Whether to flip vertically
        
    Raises:
        ValueError: If image dimensions are smaller than crop size
    """
    _, h, w = tensor.shape
    if h < size or w < size:
        raise ValueError("Image size must be >= crop size")
    if h == size and w == size:
        top = 0
        left = 0
    else:
        # Randomly sample crop coordinates
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
    # Randomly decide whether to flip
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    return top, left, flip_h, flip_v


def _apply_geom(
    tensor: torch.Tensor, size: int, top: int, left: int, flip_h: bool, flip_v: bool
) -> torch.Tensor:
    """
    Apply geometric transformations to a tensor based on sampled parameters.
    
    This function crops and optionally flips an image tensor according to the
    parameters sampled by _sample_geom_params.
    
    Args:
        tensor (torch.Tensor): Input image tensor of shape (C, H, W)
        size (int): Size of the crop
        top (int): Top coordinate for cropping
        left (int): Left coordinate for cropping
        flip_h (bool): Whether to flip horizontally
        flip_v (bool): Whether to flip vertically
        
    Returns:
        torch.Tensor: Transformed tensor of shape (C, size, size)
    """
    # Crop the tensor to the specified region
    view = tensor[:, top : top + size, left : left + size]
    # Apply horizontal flip if requested
    if flip_h:
        view = torch.flip(view, dims=[2])
    # Apply vertical flip if requested
    if flip_v:
        view = torch.flip(view, dims=[1])
    return view


def _augment_photometric(ir: torch.Tensor) -> torch.Tensor:
    """
    Apply photometric augmentations to an infrared image.
    
    This function applies various photometric transformations to increase
    the diversity of training data. These transformations simulate variations
    in lighting, camera settings, and other imaging conditions.
    
    Args:
        ir (torch.Tensor): Input infrared image tensor, values in [0, 1] range
        
    Returns:
        torch.Tensor: Augmented infrared image tensor, values in [0, 1] range
    """
    view = ir
    # Random gain (brightness adjustment)
    gain = random.uniform(0.8, 1.2)
    # Random bias (contrast adjustment)
    bias = random.uniform(-0.1, 0.1)
    # Random gamma correction
    gamma = random.uniform(0.8, 1.2)
    # Apply gain and bias, then clamp to valid range
    view = torch.clamp(view * gain + bias, 0.0, 1.0)
    # Apply gamma correction
    view = torch.pow(view, gamma)
    # Add random noise
    noise = torch.randn_like(view) * 0.02
    # Clamp final result to valid range
    view = torch.clamp(view + noise, 0.0, 1.0)
    return view


def _prepare_batch(ir: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare two augmented views of a batch of infrared images.
    
    This function implements the core of the two-view training strategy by
    creating two augmented versions of each image in the batch. Both views
    are created using the same geometric transformations but different
    photometric augmentations.
    
    Args:
        ir (torch.Tensor): Batch of infrared images of shape (B, C, H, W)
        size (int): Target size for cropping
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two augmented views of the input batch
    """
    v1_list = []
    v2_list = []
    for sample in ir:
        # Sample geometric parameters once for both views
        top, left, flip_h, flip_v = _sample_geom_params(sample, size)
        # Apply the same geometric transformation to both views
        base = _apply_geom(sample, size, top, left, flip_h, flip_v)
        # Apply different photometric augmentations to each view
        v1_list.append(_augment_photometric(base))
        v2_list.append(_augment_photometric(base))
    # Stack the views into batches
    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)
    return v1, v2


def _prune_checkpoints(checkpoint_dir: str, max_keep: int) -> None:
    """
    Keep only the most recent checkpoints (by modification time).
    """
    if max_keep <= 0:
        return
    try:
        names = os.listdir(checkpoint_dir)
    except FileNotFoundError:
        return
    candidates = []
    for name in names:
        if not name.startswith("step_") or not name.endswith(".pt"):
            continue
        path = os.path.join(checkpoint_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        candidates.append((mtime, path))
    if len(candidates) <= max_keep:
        return
    candidates.sort(key=lambda item: item[0])
    for _, path in candidates[:-max_keep]:
        try:
            os.remove(path)
        except OSError:
            pass


def _save_visualization(
    output_dir: str,
    step: int,
    ir: torch.Tensor,
    ir_hat: torch.Tensor,
    temperature: torch.Tensor,
    emissivity: torch.Tensor,
) -> None:
    """
    Save visualization of training results.
    
    This function creates visualizations of the input infrared image, reconstructed
    image, predicted temperature, and predicted emissivity. If PIL is available,
    it creates an image; otherwise, it saves the tensors in a .pt file.
    
    Args:
        output_dir (str): Directory to save the visualization
        step (int): Current training step number
        ir (torch.Tensor): Original infrared image
        ir_hat (torch.Tensor): Reconstructed infrared image
        temperature (torch.Tensor): Predicted temperature map
        emissivity (torch.Tensor): Predicted emissivity map
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        # If PIL is not available, save tensors directly
        torch.save(
            {
                "ir": ir.cpu(),
                "ir_hat": ir_hat.cpu(),
                "temperature": temperature.cpu(),
                "emissivity": emissivity.cpu(),
            },
            os.path.join(output_dir, f"step_{step}.pt"),
        )
        return

    def _to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to PIL image.
        
        Args:
            tensor (torch.Tensor): Input tensor to convert
            
        Returns:
            PIL.Image: Converted grayscale image
        """
        data = tensor.detach().cpu()
        # Normalize to [0, 1] range
        data = data - data.min()
        data = data / (data.max() + 1e-6)
        # Convert to 8-bit values
        data = (data * 255.0).byte().squeeze(0)
        return Image.fromarray(data.numpy(), mode="L")

    # Convert all tensors to PIL images
    images = [
        _to_pil(ir[0]),        # Original IR
        _to_pil(ir_hat[0]),    # Reconstructed IR
        _to_pil(temperature[0]),  # Temperature
        _to_pil(emissivity[0]),   # Emissivity
    ]
    # Calculate total width and maximum height for the grid
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    # Create a grid image combining all results
    grid = Image.new("L", (total_width, max_height))
    offset = 0
    for img in images:
        grid.paste(img, (offset, 0))
        offset += img.size[0]
    grid.save(os.path.join(output_dir, f"step_{step}.png"))


def _validate_config(config: Dict[str, object]) -> None:
    """
    Validate the training configuration dictionary.
    
    This function checks that the required keys are present in the configuration
    and that they have the expected types.
    
    Args:
        config (Dict[str, object]): Training configuration dictionary
        
    Raises:
        KeyError: If required keys are missing
        ValueError: If loss_weights is not a dictionary
    """
    # List of required keys in the configuration
    required = ["lr", "batch_size", "image_size", "loss_weights"]
    for key in required:
        if key not in config:
            raise KeyError(f"train config missing '{key}'")
    if not isinstance(config["loss_weights"], dict):
        raise ValueError("'loss_weights' must be a mapping")
    if "dataset" not in config:
        raise KeyError("train config missing 'dataset'")
    device_value = config.get("device")
    if device_value is not None and not isinstance(device_value, (str, int)):
        raise ValueError("'device' must be a string like 'cuda:0' or 'cpu'")
    lr_schedule = config.get("lr_schedule")
    if lr_schedule is not None and not isinstance(lr_schedule, (str, dict)):
        raise ValueError("'lr_schedule' must be a string or mapping")
    if isinstance(lr_schedule, dict) and "type" in lr_schedule:
        if not isinstance(lr_schedule["type"], str):
            raise ValueError("'lr_schedule.type' must be a string")


def run_training(config_path: str, steps_override: int | None = None) -> None:
    """
    Run the training process for the physical decomposition network.
    
    This function implements the complete training pipeline for the physical
    decomposition network, including:
    - Loading and validating configuration
    - Setting up the model, optimizer, and data loader
    - Implementing the two-view training strategy
    - Computing multiple physics-inspired losses
    - Saving visualizations during training
    
    Args:
        config_path (str): Path to the training configuration file
        steps_override (int | None): Override the number of training steps if provided
    """
    # Load configuration from YAML file
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("train config must be a mapping")
    _validate_config(config)

    # Extract training parameters from configuration
    lr = float(config["lr"])
    batch_size = int(config["batch_size"])
    image_size = int(config["image_size"])
    steps = int(config.get("steps", 100))
    if steps_override is not None:
        steps = int(steps_override)
    vis_interval = int(config.get("vis_interval", 50))
    eval_interval = int(config.get("eval_interval", 0))
    eval_split = str(config.get("eval_split", "test"))
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    max_checkpoints = min(5, int(config.get("max_checkpoints", 5)))
    best_metric = str(config.get("best_metric", "total")).lower()
    best_metric_mode = str(config.get("best_metric_mode", "min")).lower()
    use_noise = bool(config.get("use_noise", True))
    t_min = float(config.get("t_min", 1.0))
    t_max = float(config.get("t_max", 10.0))
    seed = int(config.get("seed", 123))

    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Determine device (GPU if available, otherwise CPU), with config override
    device = resolve_device(config)
    
    runs_dir = os.path.join("runs", "phys_decomp")
    run_logger = TrainingRunLogger.create(runs_dir, config_path)

    # Set up dataset
    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    split = str(config.get("split", "train"))
    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=split,
        loader=load_ir_sample,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="L",
        ir_mode="L",
    )
    # Create data loader with shuffling
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    iterator: Iterator[torch.Tensor] = iter(loader)

    # Initialize the model based on configuration
    model = create_phys_decomp_model(
        config=config,
        image_size=image_size,
        t_min=t_min,
        t_max=t_max,
        use_noise=use_noise,
        device=device,
    )

    run_logger.log_model_info(model, device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = create_lr_scheduler(optimizer, config, steps)

    # Extract loss weights and set up output directory
    weights = config["loss_weights"]
    checkpoint_dir = os.path.join(runs_dir, "checkpoints")
    if max_checkpoints < 1:
        max_checkpoints = 1
    metric_keys = {"total", "recon", "t_smooth", "eps_prior", "consistency", "corr"}
    if best_metric not in metric_keys:
        raise ValueError(
            f"best_metric must be one of {sorted(metric_keys)}, got '{best_metric}'"
        )
    if best_metric_mode not in {"min", "max"}:
        raise ValueError("best_metric_mode must be 'min' or 'max'")
    best_value = float("inf") if best_metric_mode == "min" else -float("inf")

    eval_loader = None
    if eval_interval > 0:
        eval_dataset = create_dataset(
            dataset_name,
            root=dataset_root,
            split=eval_split,
            loader=load_ir_sample,
            transform=lambda v, r: paired_transform(v, r, image_size),
            vis_mode="L",
            ir_mode="L",
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

    # Training loop
    for step in range(1, steps + 1):
        if scheduler is not None:
            scheduler.step(step)
        try:
            batch = next(iterator)
        except StopIteration:
            # Reset iterator when dataset is exhausted
            iterator = iter(loader)
            batch = next(iterator)

        if not isinstance(batch, dict) or "ir" not in batch:
            raise ValueError("Expected dataset batch with 'ir' tensor")
        
        # Move batch to device
        ir = batch["ir"].to(device)
        
        # Prepare two augmented views using the two-view training strategy
        v1, v2 = _prepare_batch(ir, image_size)
        v1 = v1.to(device)
        v2 = v2.to(device)

        # Forward pass for both views
        t1, eps1, n1, ir1_hat = model(v1)
        t2, eps2, n2, ir2_hat = model(v2)

        losses = compute_losses(
            v1,
            v2,
            t1,
            eps1,
            n1,
            ir1_hat,
            t2,
            eps2,
            n2,
            ir2_hat,
            weights,
        )
        total_loss = losses["total"]
        run_logger.log_losses(step, losses)

        # Perform optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print loss values for monitoring
        print(
            "step={step} total={total:.4f} recon={recon:.4f} t_smooth={t_smooth:.4f} "
            "eps_prior={eps_prior:.4f} consistency={consistency:.4f} corr={corr:.4f}".format(
                step=step,
                total=losses["total"].item(),
                recon=losses["recon"].item(),
                t_smooth=losses["t_smooth"].item(),
                eps_prior=losses["eps_prior"].item(),
                consistency=losses["consistency"].item(),
                corr=losses["corr"].item(),
            )
        )

        # Save visualization at specified intervals
        if vis_interval > 0 and step % vis_interval == 0:
            _save_visualization(runs_dir, step, v1, ir1_hat, t1, eps1)

        if eval_interval > 0 and eval_loader is not None and step % eval_interval == 0:
            metrics = evaluate_model(model, eval_loader, device, weights)
            print(
                "eval step={step} total={total:.4f} recon={recon:.4f} t_smooth={t_smooth:.4f} "
                "eps_prior={eps_prior:.4f} consistency={consistency:.4f} corr={corr:.4f}".format(
                    step=step,
                    total=metrics["total"],
                    recon=metrics["recon"],
                    t_smooth=metrics["t_smooth"],
                    eps_prior=metrics["eps_prior"],
                    consistency=metrics["consistency"],
                    corr=metrics["corr"],
                )
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "best_metric": best_metric,
                "best_metric_mode": best_metric_mode,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
            torch.save(checkpoint, checkpoint_path)
            _prune_checkpoints(checkpoint_dir, max_checkpoints)

            current_value = metrics[best_metric]
            is_better = (
                current_value < best_value
                if best_metric_mode == "min"
                else current_value > best_value
            )
            if is_better:
                best_value = current_value
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(checkpoint, best_path)
                print(
                    "best model updated: {metric}={value:.4f} at step={step}".format(
                        metric=best_metric,
                        value=best_value,
                        step=step,
                    )
                )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Minimal PhysDecompNet training")
    parser.add_argument(
        "--config",
        default="configs/phys_decomp.yml",
        help="Path to training config",
    )
    parser.add_argument("--steps", type=int, help="Override number of steps")
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the training process.
    """
    args = parse_args()
    run_training(args.config, steps_override=args.steps)


if __name__ == "__main__":
    main()
