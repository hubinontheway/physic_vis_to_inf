from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from losses.phys_losses import (
    consistency_loss,
    corr_loss,
    emissivity_prior_loss,
    recon_loss,
    smoothness_loss,
)
from datasets import create_dataset
from models.phys_decomp import PhysDecompNet
from models.vit_unet_phys import PhysDecompViTUNet
from utils.config import load_yaml

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


def _compute_losses(
    v1: torch.Tensor,
    v2: torch.Tensor,
    t1: torch.Tensor,
    eps1: torch.Tensor,
    n1: Optional[torch.Tensor],
    ir1_hat: torch.Tensor,
    t2: torch.Tensor,
    eps2: torch.Tensor,
    n2: Optional[torch.Tensor],
    ir2_hat: torch.Tensor,
    weights: Dict[str, object],
) -> Dict[str, torch.Tensor]:
    """
    Compute the weighted loss dictionary for two-view training/evaluation.
    
    Args:
        v1 (torch.Tensor): Input view 1
        v2 (torch.Tensor): Input view 2
        t1 (torch.Tensor): Temperature prediction for view 1
        eps1 (torch.Tensor): Emissivity prediction for view 1
        n1 (Optional[torch.Tensor]): Noise prediction for view 1
        ir1_hat (torch.Tensor): Reconstructed IR for view 1
        t2 (torch.Tensor): Temperature prediction for view 2
        eps2 (torch.Tensor): Emissivity prediction for view 2
        n2 (Optional[torch.Tensor]): Noise prediction for view 2
        ir2_hat (torch.Tensor): Reconstructed IR for view 2
        weights (Dict[str, object]): Loss weight mapping
        
    Returns:
        Dict[str, torch.Tensor]: Loss components including total
    """
    loss_recon = recon_loss(ir1_hat, v1) + recon_loss(ir2_hat, v2)
    loss_t_smooth = smoothness_loss(t1) + smoothness_loss(t2)
    loss_eps_prior = emissivity_prior_loss(eps1) + emissivity_prior_loss(eps2)
    loss_consistency = consistency_loss(t1, t2, eps1, eps2, n1, n2)
    loss_corr = corr_loss(t1, eps1) + corr_loss(t2, eps2)

    total_loss = (
        float(weights.get("recon", 1.0)) * loss_recon
        + float(weights.get("t_smooth", 0.1)) * loss_t_smooth
        + float(weights.get("eps_prior", 0.1)) * loss_eps_prior
        + float(weights.get("consistency", 0.1)) * loss_consistency
        + float(weights.get("corr", 0.05)) * loss_corr
    )
    return {
        "total": total_loss,
        "recon": loss_recon,
        "t_smooth": loss_t_smooth,
        "eps_prior": loss_eps_prior,
        "consistency": loss_consistency,
        "corr": loss_corr,
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: Dict[str, object],
) -> Dict[str, float]:
    """
    Run evaluation on a dataloader and return averaged metrics.
    
    Args:
        model (nn.Module): Model to evaluate
        loader (DataLoader): Evaluation dataloader
        device (torch.device): Device to run on
        weights (Dict[str, object]): Loss weight mapping
        
    Returns:
        Dict[str, float]: Averaged metrics over the dataset
    """
    was_training = model.training
    model.eval()
    totals = {
        "total": 0.0,
        "recon": 0.0,
        "t_smooth": 0.0,
        "eps_prior": 0.0,
        "consistency": 0.0,
        "corr": 0.0,
    }
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict) or "ir" not in batch:
                raise ValueError("Expected dataset batch with 'ir' tensor")
            ir = batch["ir"].to(device)
            v1 = ir
            v2 = ir
            t1, eps1, n1, ir1_hat = model(v1)
            t2, eps2, n2, ir2_hat = model(v2)
            losses = _compute_losses(
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
            batch_size = int(ir.shape[0])
            for key in totals:
                totals[key] += losses[key].item() * batch_size
            total_samples += batch_size
    if was_training:
        model.train()
    if total_samples == 0:
        raise ValueError("Evaluation loader returned no samples")
    return {key: value / total_samples for key, value in totals.items()}


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


def _pil_to_tensor(image) -> torch.Tensor:
    """
    Convert a PIL image to a PyTorch tensor.
    
    This function provides a fallback method for converting PIL images to tensors
    without requiring NumPy.
    
    Args:
        image: PIL image object
        
    Returns:
        torch.Tensor: Converted tensor normalized to [0, 1] range
    """
    try:
        import numpy as np
    except ImportError:
        # Fallback implementation without NumPy
        data = torch.tensor(list(image.getdata()), dtype=torch.float32)
        data = data.view(image.size[1], image.size[0])
        return data.unsqueeze(0) / 255.0
    # NumPy-based implementation
    array = np.array(image, dtype="float32")
    if array.ndim == 2:
        return torch.from_numpy(array).unsqueeze(0) / 255.0
    raise ValueError("Expected grayscale image")


def _load_ir_sample(path: str, mode: Optional[str] = None):
    """
    Load an infrared image sample from path.
    
    This function can load both image files and tensor files, providing
    flexibility in the data format.
    
    Args:
        path (str): Path to the image or tensor file
        mode (str, optional): PIL mode to convert to (ignored in this function)
        
    Returns:
        torch.Tensor: Loaded image as a tensor normalized to [0, 1] range
        
    Raises:
        ImportError: If trying to load image files without PIL
        ValueError: If tensor file doesn't contain a tensor
    """
    _ = mode  # mode parameter is ignored
    if path.endswith(".pt"):
        # Load tensor file
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}")
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.float()
    # Load image file
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load image files") from exc
    with Image.open(path) as img:
        return img.convert("L")


def _paired_transform(vis, ir, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform visible and infrared images to specified size.
    
    This function ensures that both visible and infrared images are resized
    to the target image size, handling both PIL images and tensor inputs.
    
    Args:
        vis: Visible image (PIL Image or tensor)
        ir: Infrared image (PIL Image or tensor)
        image_size (int): Target size for both images
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Resized visible and infrared tensors
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    # Process infrared image
    if Image is not None and isinstance(ir, Image.Image):
        # If ir is a PIL image, resize and convert to tensor
        if ir.size != (image_size, image_size):
            ir = ir.resize((image_size, image_size), Image.BILINEAR)
        ir_tensor = _pil_to_tensor(ir)
    else:
        # If ir is already a tensor, interpolate to target size
        ir_tensor = ir
        if ir_tensor.shape[-2:] != (image_size, image_size):
            ir_tensor = nn.functional.interpolate(
                ir_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    # Process visible image
    if Image is not None and isinstance(vis, Image.Image):
        # If vis is a PIL image, resize and convert to tensor
        if vis.size != (image_size, image_size):
            vis = vis.resize((image_size, image_size), Image.BILINEAR)
        vis_tensor = _pil_to_tensor(vis)
    else:
        # If vis is already a tensor, interpolate to target size
        vis_tensor = vis
        if vis_tensor.shape[-2:] != (image_size, image_size):
            vis_tensor = nn.functional.interpolate(
                vis_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    return vis_tensor, ir_tensor


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


def _resolve_device(config: Dict[str, object]) -> torch.device:
    """
    Resolve the training device from configuration.

    Args:
        config (Dict[str, object]): Training configuration dictionary

    Returns:
        torch.device: Selected device for training
    """
    device_value = config.get("device")
    if device_value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device_value, int):
        device_str = f"cuda:{device_value}"
    else:
        device_str = str(device_value)

    try:
        device = torch.device(device_str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid device '{device_value}'") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available")
        if device.index is not None:
            count = torch.cuda.device_count()
            if device.index < 0 or device.index >= count:
                raise ValueError(
                    f"CUDA device index {device.index} out of range "
                    f"(device_count={count})"
                )
            torch.cuda.set_device(device.index)
    return device


def _ensure_timm_available() -> None:
    """
    Ensure that the timm library is available.
    
    This function checks if timm is installed and raises an ImportError if not.
    It's used when the ViT-UNet model is configured to use timm models.
    
    Raises:
        ImportError: If timm is not installed
    """
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "timm is required for model_type=vit_unet with use_timm=true. "
            "Install with `pip install timm`."
        ) from exc


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
    device = _resolve_device(config)
    
    # Set up dataset
    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    split = str(config.get("split", "train"))
    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=split,
        loader=_load_ir_sample,
        transform=lambda v, r: _paired_transform(v, r, image_size),
        vis_mode="L",
        ir_mode="L",
    )
    # Create data loader with shuffling
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    iterator: Iterator[torch.Tensor] = iter(loader)

    # Initialize the model based on configuration
    model_type = str(config.get("model_type", "cnn")).lower()
    if model_type == "vit_unet":
        vit_cfg = config.get("vit_unet", {})
        if not isinstance(vit_cfg, dict):
            raise ValueError("'vit_unet' must be a mapping")
        if bool(vit_cfg.get("use_timm", False)):
            _ensure_timm_available()
        model = PhysDecompViTUNet(
            in_channels=1,
            base_channels=int(vit_cfg.get("base_channels", 32)),
            vit_embed_dim=int(vit_cfg.get("vit_embed_dim", 256)),
            vit_depth=int(vit_cfg.get("vit_depth", 6)),
            vit_heads=int(vit_cfg.get("vit_heads", 8)),
            patch_size=int(vit_cfg.get("patch_size", 16)),
            image_size=image_size,
            t_min=t_min,
            t_max=t_max,
            use_noise=use_noise,
            use_mlp=bool(vit_cfg.get("use_mlp", False)),
            use_timm=bool(vit_cfg.get("use_timm", False)),
            timm_model=str(vit_cfg.get("timm_model", "vit_base_patch16_224")),
            timm_pretrained=bool(vit_cfg.get("timm_pretrained", False)),
        ).to(device)
    else:
        model = PhysDecompNet(
            in_channels=1,
            base_channels=32,
            t_min=t_min,
            t_max=t_max,
            use_noise=use_noise,
        ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Extract loss weights and set up output directory
    weights = config["loss_weights"]
    runs_dir = os.path.join("runs", "phys_decomp")
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
            loader=_load_ir_sample,
            transform=lambda v, r: _paired_transform(v, r, image_size),
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

        losses = _compute_losses(
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
            metrics = _evaluate_model(model, eval_loader, device, weights)
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
