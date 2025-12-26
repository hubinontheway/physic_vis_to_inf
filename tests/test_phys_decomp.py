import os
import tempfile

import torch

from models.phys_decomp import PhysDecompNet
from models.physics_layer import PhysicsLayer
from train_phys_decomp import run_training


def test_forward_shapes():
    model = PhysDecompNet(in_channels=1, use_noise=True)
    x = torch.randn(2, 1, 32, 32)
    t, eps, n, ir_hat = model(x)
    assert t.shape == (2, 1, 32, 32)
    assert eps.shape == (2, 1, 32, 32)
    assert n is not None and n.shape == (2, 1, 32, 32)
    assert ir_hat.shape == (2, 1, 32, 32)


def test_physics_layer_backward():
    layer = PhysicsLayer()
    t = torch.randn(1, 1, 8, 8, requires_grad=True)
    eps = torch.sigmoid(torch.randn(1, 1, 8, 8, requires_grad=True))
    n = torch.randn(1, 1, 8, 8, requires_grad=True)
    ir_hat = layer(t, eps, n)
    loss = ir_hat.mean()
    loss.backward()


def test_training_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = os.path.join(tmpdir, "VEDIA")
        os.makedirs(os.path.join(dataset_root, "trainA"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "trainB"), exist_ok=True)
        sample_ir_path = os.path.join(dataset_root, "trainB", "sample.pt")
        sample_vis_path = os.path.join(dataset_root, "trainA", "sample.pt")
        torch.save(torch.rand(1, 16, 16), sample_ir_path)
        torch.save(torch.rand(1, 16, 16), sample_vis_path)

        config_path = os.path.join(tmpdir, "phys_decomp_vit_no_pretrain.yml")
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(
                "lr: 0.0001\n"
                "batch_size: 2\n"
                "image_size: 16\n"
                "steps: 2\n"
                "vis_interval: 0\n"
                "dataset: VEDIA\n"
                f"dataset_root: {dataset_root}\n"
                "loss_weights:\n"
                "  recon: 1.0\n"
                "  t_smooth: 0.1\n"
                "  eps_prior: 0.1\n"
                "  consistency: 0.1\n"
                "  corr: 0.05\n"
            )

        run_training(config_path, steps_override=2)
