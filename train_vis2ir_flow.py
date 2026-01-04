from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from flow_matching.path import CondOTProbPath
from datasets import create_dataset
from models.phys_decomp_factory import create_phys_decomp_model
from models.vis2ir_flow import Vis2IRFlowUNet
from models.vis2phys_align import Vis2PhysAlignUNet
from utils.config import load_yaml
from utils.device import resolve_device
from utils.flow_sampling import build_solver, sample_ir
from utils.lr_schedule import create_lr_scheduler
from utils.metrics import psnr, ssim
from utils.run_artifacts import find_best_checkpoint, find_latest_config
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
    phys_align = config.get("phys_align", {})
    if phys_align is not None and not isinstance(phys_align, dict):
        raise ValueError("'phys_align' must be a mapping")
    phys_align = phys_align or {}
    cond_mode = str(phys_align.get("cond_mode", "vis")).lower()
    if cond_mode not in {"vis", "phys", "vis_phys"}:
        raise ValueError("phys_align.cond_mode must be 'vis', 'phys', or 'vis_phys'")
    if not bool(phys_align.get("enabled", False)) and cond_mode != "vis":
        raise ValueError("phys_align.cond_mode requires phys_align.enabled=true")


def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)


def _load_phys_teacher(
    teacher_cfg: Dict[str, object],
    device: torch.device,
    fallback_image_size: int,
) -> torch.nn.Module:
    run_dir = teacher_cfg.get("run_dir")
    config_path = teacher_cfg.get("config_path")
    if config_path is None and run_dir:
        config_path, _ = find_latest_config(str(run_dir))
    if config_path is None:
        raise ValueError("phys_align.teacher requires config_path or run_dir")

    config = load_yaml(str(config_path))
    if not isinstance(config, dict):
        raise ValueError("phys_align.teacher config must be a mapping")

    image_size_value = teacher_cfg.get("image_size")
    if image_size_value is None:
        image_size_value = config.get("image_size", fallback_image_size)
    image_size = int(image_size_value)

    t_min_value = teacher_cfg.get("t_min")
    if t_min_value is None:
        t_min_value = config.get("t_min", 1.0)
    t_min = float(t_min_value)

    t_max_value = teacher_cfg.get("t_max")
    if t_max_value is None:
        t_max_value = config.get("t_max", 10.0)
    t_max = float(t_max_value)

    use_noise_value = teacher_cfg.get("use_noise")
    if use_noise_value is None:
        use_noise_value = config.get("use_noise", True)
    use_noise = bool(use_noise_value)

    model = create_phys_decomp_model(
        config=config,
        image_size=image_size,
        t_min=t_min,
        t_max=t_max,
        use_noise=use_noise,
        device=device,
    )

    checkpoint_path = teacher_cfg.get("checkpoint")
    if checkpoint_path is None and run_dir:
        checkpoint_path = find_best_checkpoint(str(run_dir))
    if checkpoint_path is None:
        raise ValueError("phys_align.teacher requires checkpoint or run_dir")

    _load_checkpoint(model, str(checkpoint_path), device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


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


def _build_cond(
    vis: torch.Tensor,
    phys_align: Optional[Vis2PhysAlignUNet],
    cond_mode: str,
    t_cond_norm: bool,
    return_phys: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]] | torch.Tensor:
    if cond_mode == "vis":
        if return_phys:
            return vis, None, None
        return vis
    if phys_align is None:
        raise ValueError("phys_align must be enabled for cond_mode != 'vis'")
    temperature, emissivity = phys_align(vis)
    temperature_cond = temperature
    if t_cond_norm:
        denom = float(phys_align.t_max - phys_align.t_min)
        temperature_cond = (temperature - float(phys_align.t_min)) / denom
        temperature_cond = torch.clamp(temperature_cond, 0.0, 1.0)
    if cond_mode == "phys":
        cond = torch.cat([temperature_cond, emissivity], dim=1)
        if return_phys:
            return cond, temperature, emissivity
        return cond
    if cond_mode == "vis_phys":
        cond = torch.cat([vis, temperature_cond, emissivity], dim=1)
        if return_phys:
            return cond, temperature, emissivity
        return cond
    raise ValueError(f"Unsupported cond_mode '{cond_mode}'")


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

    device = resolve_device(config)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    runs_dir = os.path.join("runs", config_name)

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

    phys_align_cfg = config.get("phys_align", {}) or {}
    phys_align_enabled = bool(phys_align_cfg.get("enabled", False))
    cond_mode = str(phys_align_cfg.get("cond_mode", "vis")).lower()
    t_cond_norm = bool(phys_align_cfg.get("t_cond_norm", True))
    teacher_cfg = phys_align_cfg.get("teacher", {}) or {}
    if teacher_cfg is not None and not isinstance(teacher_cfg, dict):
        raise ValueError("'phys_align.teacher' must be a mapping")
    teacher_enabled = bool(teacher_cfg.get("enabled", False))
    if teacher_enabled and not phys_align_enabled:
        raise ValueError("phys_align.teacher requires phys_align.enabled=true")
    if cond_mode == "vis":
        cond_channels = 3
    elif cond_mode == "phys":
        cond_channels = 2
    else:
        cond_channels = 5

    flow_model_cfg = config.get("flow_model", {}) or {}
    base_channels = int(flow_model_cfg.get("base_channels", 32))
    channel_mults = flow_model_cfg.get("channel_mults", [1, 2, 4])
    if not isinstance(channel_mults, (list, tuple)):
        raise ValueError("'flow_model.channel_mults' must be a list of ints")
    channel_mults = tuple(int(v) for v in channel_mults)
    model = Vis2IRFlowUNet(
        in_channels=1,
        cond_channels=cond_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
    ).to(device)
    phys_align = None
    if phys_align_enabled:
        phys_base_channels = int(phys_align_cfg.get("base_channels", 16))
        phys_mults = phys_align_cfg.get("channel_mults", [1, 2, 4])
        if not isinstance(phys_mults, (list, tuple)):
            raise ValueError("'phys_align.channel_mults' must be a list of ints")
        phys_mults = tuple(int(v) for v in phys_mults)
        phys_t_min = float(phys_align_cfg.get("t_min", 1.0))
        phys_t_max = float(phys_align_cfg.get("t_max", 10.0))
        phys_align = Vis2PhysAlignUNet(
            in_channels=3,
            base_channels=phys_base_channels,
            channel_mults=phys_mults,
            t_min=phys_t_min,
            t_max=phys_t_max,
        ).to(device)

    phys_teacher = None
    teacher_loss_weight = 0.0
    teacher_t_weight = 1.0
    teacher_eps_weight = 1.0
    if teacher_enabled:
        phys_teacher = _load_phys_teacher(teacher_cfg, device, image_size)
        teacher_loss_weight = float(teacher_cfg.get("loss_weight", 1.0))
        teacher_t_weight = float(teacher_cfg.get("t_weight", 1.0))
        teacher_eps_weight = float(teacher_cfg.get("eps_weight", 1.0))

    loss_columns = ("step", "loss", "lr")
    if phys_teacher is not None:
        loss_columns = ("step", "loss", "phys_teacher", "lr")
    run_logger = TrainingRunLogger.create(
        runs_dir,
        config_path,
        loss_columns=loss_columns,
    )

    log_model = model
    if phys_align is not None:
        log_model = nn.ModuleDict({"flow": model, "phys_align": phys_align})
    run_logger.log_model_info(log_model, device)

    params = list(model.parameters())
    if phys_align is not None:
        params += list(phys_align.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
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
    solver = build_solver(model)
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
        cond_result = _build_cond(
            vis,
            phys_align,
            cond_mode,
            t_cond_norm,
            return_phys=phys_teacher is not None,
        )
        if phys_teacher is not None:
            cond, phys_t_pred, phys_eps_pred = cond_result
        else:
            cond = cond_result
        pred = model(path_sample.x_t, t, cond=cond)
        flow_loss = nn.functional.mse_loss(pred, path_sample.dx_t)
        phys_teacher_loss = None
        if phys_teacher is not None:
            if phys_t_pred is None or phys_eps_pred is None:
                phys_t_pred, phys_eps_pred = phys_align(vis)
            with torch.no_grad():
                teacher_t, teacher_eps, _, _ = phys_teacher(ir)
            t_loss = F.l1_loss(phys_t_pred, teacher_t)
            eps_loss = F.l1_loss(phys_eps_pred, teacher_eps)
            phys_teacher_loss = teacher_loss_weight * (
                teacher_t_weight * t_loss + teacher_eps_weight * eps_loss
            )
        loss = flow_loss if phys_teacher_loss is None else flow_loss + phys_teacher_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        log_payload = {"loss": loss.item(), "lr": current_lr}
        if phys_teacher_loss is not None:
            log_payload["phys_teacher"] = phys_teacher_loss.item()
        run_logger.log_losses(step, log_payload)
        print(f"step={step} loss={loss.item():.6f} lr={current_lr:.6f}")

        if vis_interval > 0 and step % vis_interval == 0:
            with torch.no_grad():
                cond = _build_cond(vis, phys_align, cond_mode, t_cond_norm)
                pred_ir = sample_ir(solver, cond, sampling_cfg).clamp(0.0, 1.0)
            _save_visualization(runs_dir, step, vis, ir, pred_ir)

        if eval_interval > 0 and eval_loader is not None and step % eval_interval == 0:
            model.eval()
            if phys_align is not None:
                phys_align.eval()
            total_psnr = 0.0
            total_ssim = 0.0
            total_samples = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    eval_vis = eval_batch["vis"].to(device)
                    eval_ir = eval_batch["ir"].to(device)
                    cond = _build_cond(eval_vis, phys_align, cond_mode, t_cond_norm)
                    pred_ir = sample_ir(solver, cond, sampling_cfg).clamp(0.0, 1.0)
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
            if phys_align is not None:
                checkpoint["phys_align"] = phys_align.state_dict()
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
            if phys_align is not None:
                phys_align.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flow-matching training for visible-to-infrared translation"
    )
    parser.add_argument(
        "--config",
        default="configs/vis2ir_flow_phys_align_teacher.yml",
        help="Path to training config",
    )
    parser.add_argument("--steps", type=int, help="Override number of steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config, steps_override=args.steps)


if __name__ == "__main__":
    main()
