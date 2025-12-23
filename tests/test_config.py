import os
import tempfile
import unittest

from train import build_dataset_from_config
from utils.config import load_dataset_paths


def _dummy_loader(path, mode=None):
    return f"{mode}:{os.path.basename(path)}"


class ConfigTests(unittest.TestCase):
    def test_load_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "datasets.yml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write("datasets:\n  VEDIA: /tmp/data/VEDIA\n")

            paths = load_dataset_paths(config_path)
            self.assertEqual(paths["VEDIA"], "/tmp/data/VEDIA")

    def test_build_dataset_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "VEDIA")
            os.makedirs(os.path.join(root, "trainA"), exist_ok=True)
            os.makedirs(os.path.join(root, "trainB"), exist_ok=True)
            with open(os.path.join(root, "trainA", "0001.jpg"), "w", encoding="utf-8") as handle:
                handle.write("vis")
            with open(os.path.join(root, "trainB", "0001.jpg"), "w", encoding="utf-8") as handle:
                handle.write("ir")

            config_path = os.path.join(tmpdir, "datasets.yml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(f"datasets:\n  VEDIA: {root}\n")

            dataset = build_dataset_from_config(
                config_path,
                dataset_name="VEDIA",
                split="train",
                loader=_dummy_loader,
            )

            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
