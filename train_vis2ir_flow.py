from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from datasets import create_dataset
from models.vis2ir_flow import Vis2IRFlowUNet
from utils.config import load_yaml
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim
from utils.training_logger import TrainingRunLogger
from utils.vision import load_tensor_or_pil, paired_transform


def _validate_config(config: Dict[str, object]) -> None:
    required = ["lr", "batch_size", "image_size", "steps", "dataset"]
    for key in required:
        if key not in config:
            raise KeyError(f"train config missing '{key}'")
    device_value = config.get("device")
    if device_value is not None and not isinstance(device_value, (str, int)):
        raise ValueError("'device' must be a string like 'cuda:0' or 'cpu'")
    lr_schedule = config.get("lr_schedule")
    if lr_schedule is not None and not isinstance(lr_schedule, (str, dict)):
        raise ValueError("'lr_schedule' must be a string or mapping")
    flow_model = config.get("flow_model", {})
    if flow_model is not None and not isinstance(flow_model, dict):
        raise ValueError("'flow_model' must be a mapping")
    flow_sampling = config.get("flow_sampling", {})
    if flow_sampling is not None and not isinstance(flow_sampling, dict):
        raise ValueError("'flow_sampling' must be a mapping")


def _resolve_device(config: Dict[str, object]) -> torch.device:
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


def _prune_checkpoints(checkpoint_dir: str, max_keep: int) -> None:
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


def _tensor_to_pil(tensor: torch.Tensor, mode: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for visualization") from exc
    data = tensor.detach().cpu()
    data = torch.clamp(data, 0.0, 1.0)
    if mode == "RGB":
        data = (data * 255.0).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(data, mode="RGB")
    data = (data * 255.0).byte().squeeze(0).numpy()
    return Image.fromarray(data, mode="L")


def _save_visualization(
    output_dir: str,
    step: int,
    vis: torch.Tensor,
    ir: torch.Tensor,
    pred: torch.Tensor,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        torch.save(
            {"vis": vis.cpu(), "ir": ir.cpu(), "pred": pred.cpu()},
            os.path.join(output_dir, f"step_{step}.pt"),
        )
        return

    vis_img = _tensor_to_pil(vis[0], mode="RGB")
    ir_img = _tensor_to_pil(ir[0], mode="L").convert("RGB")
    pred_img = _tensor_to_pil(pred[0], mode="L").convert("RGB")
    width, height = vis_img.size
    grid = Image.new("RGB", (width * 3, height))
    grid.paste(vis_img, (0, 0))
    grid.paste(ir_img, (width, 0))
    grid.paste(pred_img, (width * 2, 0))
    grid.save(os.path.join(output_dir, f"step_{step}.png"))


class _ConditionalVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        cond = extras.get("cond")
        if cond is None:
            raise ValueError("cond must be provided for conditional sampling")
        return self.model(x=x, t=t, cond=cond)


def _sample_ir(
    solver: ODESolver,
    cond: torch.Tensor,
    sampling_cfg: Dict[str, object],
) -> torch.Tensor:
    steps = int(sampling_cfg.get("steps", 50))
    method = str(sampling_cfg.get("method", "euler"))
    step_size = float(sampling_cfg.get("step_size", 1.0 / max(steps, 1)))
    atol = float(sampling_cfg.get("atol", 1e-5))
    rtol = float(sampling_cfg.get("rtol", 1e-5))
    time_grid = torch.tensor([0.0, 1.0], device=cond.device)
    x_init = torch.randn(cond.shape[0], 1, cond.shape[-2], cond.shape[-1], device=cond.device)
    return solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        atol=atol,
        rtol=rtol,
        time_grid=time_grid,
        cond=cond,
    )


def run_training(config_path: str, steps_override: int | None = None) -> None:
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("train config must be a mapping")
    _validate_config(config)

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
    best_metric = str(config.get("best_metric", "psnr")).lower()
    best_metric_mode = str(config.get("best_metric_mode", "max")).lower()
    seed = int(config.get("seed", 123))

    random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(config)
    runs_dir = os.path.join("runs", "vis2ir_flow")
    run_logger = TrainingRunLogger.create(
        runs_dir,
        config_path,
        loss_columns=("step", "loss", "lr"),
    )

    dataset_name = str(config["dataset"])
    data_root = str(config.get("data_root", "/data2/hubin/datasets"))
    dataset_root = str(config.get("dataset_root", os.path.join(data_root, dataset_name)))
    split = str(config.get("split", "train"))
    dataset = create_dataset(
        dataset_name,
        root=dataset_root,
        split=split,
        loader=load_tensor_or_pil,
        transform=lambda v, r: paired_transform(v, r, image_size),
        vis_mode="RGB",
        ir_mode="L",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    iterator: Iterator[Dict[str, torch.Tensor]] = iter(loader)

    flow_model_cfg = config.get("flow_model", {}) or {}
    base_channels = int(flow_model_cfg.get("base_channels", 32))
    channel_mults = flow_model_cfg.get("channel_mults", [1, 2, 4])
    if not isinstance(channel_mults, (list, tuple)):
        raise ValueError("'flow_model.channel_mults' must be a list of ints")
    channel_mults = tuple(int(v) for v in channel_mults)
    model = Vis2IRFlowUNet(
        in_channels=1,
        cond_channels=3,
        base_channels=base_channels,
        channel_mults=channel_mults,
    ).to(device)
    run_logger.log_model_info(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = create_lr_scheduler(optimizer, config, steps)

    checkpoint_dir = os.path.join(runs_dir, "checkpoints")
    if max_checkpoints < 1:
        max_checkpoints = 1
    metric_keys = {"psnr", "ssim"}
    if best_metric not in metric_keys:
        raise ValueError(f"best_metric must be one of {sorted(metric_keys)}")
    if best_metric_mode not in {"min", "max"}:
        raise ValueError("best_metric_mode must be 'min' or 'max'")
    best_value = -float("inf") if best_metric_mode == "max" else float("inf")

    eval_loader = None
    if eval_interval > 0:
        eval_dataset = create_dataset(
            dataset_name,
            root=dataset_root,
            split=eval_split,
            loader=load_tensor_or_pil,
            transform=lambda v, r: paired_transform(v, r, image_size),
            vis_mode="RGB",
            ir_mode="L",
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

    path = CondOTProbPath()
    solver = ODESolver(_ConditionalVelocityWrapper(model))
    sampling_cfg = config.get("flow_sampling", {}) or {}

    for step in range(1, steps + 1):
        if scheduler is not None:
            scheduler.step(step)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        if not isinstance(batch, dict) or "vis" not in batch or "ir" not in batch:
            raise ValueError("Expected dataset batch with 'vis' and 'ir' tensors")
        vis = batch["vis"].to(device)
        ir = batch["ir"].to(device)
        if vis.dim() != 4 or ir.dim() != 4:
            raise ValueError("Expected 'vis' and 'ir' to be 4D tensors")

        t = torch.rand(vis.shape[0], device=device)
        x0 = torch.randn_like(ir)
        path_sample = path.sample(x_0=x0, x_1=ir, t=t)
        pred = model(path_sample.x_t, t, cond=vis)
        loss = nn.functional.mse_loss(pred, path_sample.dx_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        run_logger.log_losses(step, {"loss": loss.item(), "lr": current_lr})
        print(f"step={step} loss={loss.item():.6f} lr={current_lr:.6f}")

        if vis_interval > 0 and step % vis_interval == 0:
            with torch.no_grad():
                pred_ir = _sample_ir(solver, vis, sampling_cfg).clamp(0.0, 1.0)
            _save_visualization(runs_dir, step, vis, ir, pred_ir)

        if eval_interval > 0 and eval_loader is not None and step % eval_interval == 0:
            model.eval()
            total_psnr = 0.0
            total_ssim = 0.0
            total_samples = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    eval_vis = eval_batch["vis"].to(device)
                    eval_ir = eval_batch["ir"].to(device)
                    pred_ir = _sample_ir(solver, eval_vis, sampling_cfg).clamp(0.0, 1.0)
                    batch_psnr = psnr(pred_ir, eval_ir).mean().item()
                    batch_ssim = ssim(pred_ir, eval_ir).mean().item()
                    batch_size = int(eval_ir.shape[0])
                    total_psnr += batch_psnr * batch_size
                    total_ssim += batch_ssim * batch_size
                    total_samples += batch_size

            if total_samples == 0:
                raise ValueError("Evaluation loader returned no samples")
            metrics = {
                "psnr": total_psnr / total_samples,
                "ssim": total_ssim / total_samples,
            }
            print(
                "eval step={step} psnr={psnr:.4f} ssim={ssim:.4f}".format(
                    step=step,
                    psnr=metrics["psnr"],
                    ssim=metrics["ssim"],
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
            model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flow-matching training for visible-to-infrared translation"
    )
    parser.add_argument(
        "--config",
        default="configs/vis2ir_flow.yml",
        help="Path to training config",
    )
    parser.add_argument("--steps", type=int, help="Override number of steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config, steps_override=args.steps)


if __name__ == "__main__":
    main()
