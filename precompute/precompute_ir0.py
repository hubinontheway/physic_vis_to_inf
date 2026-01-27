from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import create_dataset
from models.vis2ir_etrl_pl import Vis2IRETRLPlModule
from utils.config import load_yaml
from utils.device import resolve_device
from utils.vision import load_tensor_or_pil, tensor_to_pil


def _pil_to_tensor(image) -> torch.Tensor:
    try:
        import numpy as np
    except ImportError:
        data = list(image.getdata())
        if image.mode == "RGB":
            tensor = torch.tensor(data, dtype=torch.float32).view(
                image.size[1], image.size[0], 3
            )
            tensor = tensor.permute(2, 0, 1)
            return tensor / 255.0
        tensor = torch.tensor(data, dtype=torch.float32).view(
            image.size[1], image.size[0]
        )
        return tensor.unsqueeze(0) / 255.0

    array = np.array(image, dtype="float32")
    if array.ndim == 2:
        return torch.from_numpy(array).unsqueeze(0) / 255.0
    if array.ndim == 3:
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0
    raise ValueError("Unsupported image array shape")


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    if tensor.numel() == 0:
        return tensor
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.max() > 1.0 or tensor.min() < 0.0:
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def _ensure_vis_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 3:
        raise ValueError("Expected vis tensor to have shape (C, H, W)")
    if tensor.shape[0] == 3:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    raise ValueError("Visible tensor must have 1 or 3 channels")


def _load_vis(path: str, image_size: int) -> torch.Tensor:
    data = load_tensor_or_pil(path, mode="RGB")
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = _pil_to_tensor(data)
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected a single image in {path}")
        tensor = tensor.squeeze(0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    tensor = _normalize_tensor(tensor)
    tensor = _ensure_vis_channels(tensor)
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def _resolve_output_dir(base_dir: str, split: str, use_split_subdir: bool) -> str:
    if use_split_subdir:
        return os.path.join(base_dir, split)
    return base_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow/vis2ir_e2f_etra.yml")
    parser.add_argument("--output-dir", required=True, help="Directory to save precomputed IR0 images.")
    parser.add_argument("--split", default=None, help="Dataset split to process (train/test).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-split-subdir", action="store_true", help="Save under output/split/")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")

    image_size = int(config.get("image_size", 256))
    dataset_name = str(config.get("dataset", "VEDIA"))
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    split = args.split or str(config.get("split", "train"))
    batch_size = args.batch_size or int(config.get("batch_size", 1))

    etrl_cfg: Dict[str, object] = config.get("etrl", {}) or {}
    etrl_checkpoint = etrl_cfg.get("checkpoint")
    if not etrl_checkpoint:
        raise ValueError("etrl.checkpoint must be provided in config")

    device = resolve_device(config)
    pl_model = Vis2IRETRLPlModule.load_from_checkpoint(etrl_checkpoint, map_location=device)
    pl_model.to(device)
    pl_model.eval()

    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=split,
        loader=load_tensor_or_pil,
        transform=None,
        vis_mode="RGB",
        ir_mode="L",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    output_dir = _resolve_output_dir(args.output_dir, split, args.use_split_subdir)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            vis_paths = batch.get("vis_path")
            ir_paths = batch.get("ir_path")
            if vis_paths is None:
                raise ValueError("Dataset must provide vis_path for preprocessing")
            vis_batch = []
            for path in vis_paths:
                vis_tensor = _load_vis(path, image_size)
                vis_batch.append(vis_tensor)
            vis = torch.stack(vis_batch, dim=0).to(device)
            pred_ir = pl_model(vis).clamp(0.0, 1.0).cpu()
            for i, path in enumerate(vis_paths):
                name_source = path
                if ir_paths is not None:
                    name_source = ir_paths[i]
                filename = os.path.basename(name_source)
                out_path = os.path.join(output_dir, filename)
                tensor_to_pil(pred_ir[i], mode="L").save(out_path)

    print(f"Saved IR0 images to {output_dir}")


if __name__ == "__main__":
    main()
