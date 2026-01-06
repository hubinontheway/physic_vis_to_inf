from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from datasets import create_dataset
from models.etra_pl import ETRAPlModule
from utils.config import load_yaml
from utils.vision import load_tensor_or_pil, paired_transform, tensor_to_pil
from utils.run_artifacts import find_best_checkpoint, find_latest_config, save_eval_metrics
from utils.metrics import psnr, ssim


class IRWrapper(Dataset):
    """Wraps PairedImageDataset to return only IR images as 'image' key."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return {"image": data["ir"]}


def _resolve_run_dir(run_dir: str | None, config_path: str | None) -> str:
    if run_dir:
        return run_dir
    if config_path:
        config_dir = os.path.dirname(config_path)
        name = os.path.basename(config_path)
        if name.startswith("config_") and name.endswith(".yml") and config_dir:
            return config_dir
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        return os.path.join("runs", config_name)
    raise ValueError("Either --run-dir or --config must be provided")


def run_eval(run_dir: str) -> Dict[str, float]:
    # 1. Load config
    config_path, run_id = find_latest_config(run_dir)
    config = load_yaml(config_path)

    # 2. Setup Device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and ":" not in str(device):
        device = torch.device("cuda:0")

    # 3. Load Dataset
    dataset_name = str(config.get("dataset", "VEDIA"))
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    image_size = int(config.get("image_size", 256))
    eval_split = str(config.get("eval_split", "test"))
    batch_size = int(config.get("batch_size", 1))

    dataset_base = create_dataset(
        dataset_name,
        root=dataset_root,
        split=eval_split,
        loader=load_tensor_or_pil,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="RGB",
        ir_mode="L",
    )
    dataset = IRWrapper(dataset_base)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 4. Load Model
    checkpoint_path = find_best_checkpoint(run_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # We might need to map keys if config structure changed, but usually load_from_checkpoint is robust 
    # if hparams were saved. However, let's pass config params just in case.
    model_params = config.get("model_params", {})
    pl_model = ETRAPlModule.load_from_checkpoint(checkpoint_path, **model_params)
    pl_model.to(device)
    pl_model.eval()

    # 5. Evaluation Loop
    save_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    
    sample_rows = []
    sample_index = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Batch is {"image": tensor}
            x = batch["image"].to(device)
            
            start_time = time.perf_counter()
            
            # ETRAPlModule logic
            # _get_input handles rearrangement if needed, but our loader gives (B, 1, H, W)
            # x is [-1, 1]. _to_01 converts to [0, 1].
            i = pl_model._to_01(x)
            preds = pl_model(i)
            decoded = pl_model._decode(preds)
            i_hat, _, _, _ = pl_model._reconstruct(decoded)
            
            # Clamp i_hat to 0..1 for metrics
            i_hat = i_hat.clamp(0.0, 1.0)
            
            elapsed = time.perf_counter() - start_time
            
            # Metrics
            batch_psnr = psnr(i_hat, i).mean().item()
            batch_ssim = ssim(i_hat, i).mean().item()
            
            b_size = x.size(0)
            total_psnr += batch_psnr * b_size
            total_ssim += batch_ssim * b_size
            total_samples += b_size
            
            per_image = elapsed / max(b_size, 1)

            # Save Images
            # We save: Input, Recon, Components
            # Components: T, R, A, B, Eps, Tau
            
            i_cpu = i.cpu()
            i_hat_cpu = i_hat.cpu()
            t_cpu = pl_model._minmax_norm(decoded["t"]).cpu()
            r_cpu = decoded["r"].cpu() # 0..1
            a_cpu = decoded["a"].cpu() # 0..1
            
            for k in range(b_size):
                idx = sample_index + k
                base_name = f"sample_{idx:06d}"
                
                # Save composite grid
                # Row: Input, Recon, T, R, A
                row = torch.cat([i_cpu[k], i_hat_cpu[k], t_cpu[k], r_cpu[k], a_cpu[k]], dim=2)
                save_image(row, os.path.join(save_dir, f"{base_name}_grid.png"))
                
                # Compute per-sample metrics for CSV
                p_val = psnr(i_hat[k:k+1], i[k:k+1]).item()
                s_val = ssim(i_hat[k:k+1], i[k:k+1]).item()
                
                sample_rows.append({
                    "index": idx,
                    "filename": f"{base_name}_grid.png",
                    "psnr": p_val,
                    "ssim": s_val,
                    "inference_time": per_image
                })
            
            sample_index += b_size

    if total_samples == 0:
        print("Warning: No samples found.")
        return {}
        
    avg_metrics = {
        "psnr": total_psnr / total_samples, 
        "ssim": total_ssim / total_samples
    }
    
    print(f"Evaluation Complete. PSNR: {avg_metrics['psnr']:.4f}, SSIM: {avg_metrics['ssim']:.4f}")
    save_eval_metrics(run_dir, run_id, avg_metrics, split=eval_split)
    
    # Save CSV
    if sample_rows:
        with open(os.path.join(save_dir, "eval_stats.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "filename", "psnr", "ssim", "inference_time"])
            writer.writeheader()
            writer.writerows(sample_rows)
            
    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    args = parser.parse_args()
    
    run_dir = _resolve_run_dir(args.run_dir, args.config)
    run_eval(run_dir)


if __name__ == "__main__":
    main()
