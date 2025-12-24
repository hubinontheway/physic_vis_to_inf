import os
import tempfile
import unittest

from train import build_dataset_from_train_config
from utils.config import load_train_config


def _dummy_loader(path, mode=None):
    return f"{mode}:{os.path.basename(path)}"


class ConfigTests(unittest.TestCase):
    def test_load_train_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "train.yml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write("dataset: VEDIA\nsplit: train\n")

            config = load_train_config(config_path)
            self.assertEqual(config["dataset"], "VEDIA")

    def test_build_dataset_from_train_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "VEDIA")
            os.makedirs(os.path.join(root, "trainA"), exist_ok=True)
            os.makedirs(os.path.join(root, "trainB"), exist_ok=True)
            with open(os.path.join(root, "trainA", "0001.jpg"), "w", encoding="utf-8") as handle:
                handle.write("vis")
            with open(os.path.join(root, "trainB", "0001.jpg"), "w", encoding="utf-8") as handle:
                handle.write("ir")

            config_path = os.path.join(tmpdir, "train.yml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(f"dataset: VEDIA\ndataset_root: {root}\n")

            dataset = build_dataset_from_train_config(
                config_path,
                loader=_dummy_loader,
            )

            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
