import os
import tempfile
import unittest

from datasets import create_dataset


def _make_dataset_root(tmpdir):
    root = os.path.join(tmpdir, "VEDIA")
    for split in ("train", "test"):
        os.makedirs(os.path.join(root, f"{split}A"), exist_ok=True)
        os.makedirs(os.path.join(root, f"{split}B"), exist_ok=True)
    return root


def _write_pair(root, split, name):
    for suffix in ("A", "B"):
        path = os.path.join(root, f"{split}{suffix}", name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{split}{suffix}:{name}")


def _dummy_loader(path, mode=None):
    return f"{mode}:{os.path.basename(path)}"


class PairedDatasetTests(unittest.TestCase):
    def test_pairs_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset_root(tmpdir)
            _write_pair(root, "train", "0001.jpg")
            _write_pair(root, "train", "0002.jpg")

            dataset = create_dataset(
                "VEDIA",
                root=root,
                split="train",
                loader=_dummy_loader,
            )

            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertIn("vis", sample)
            self.assertIn("ir", sample)
            self.assertTrue(sample["vis"].startswith("RGB:"))
            self.assertTrue(sample["ir"].startswith("L:"))

    def test_mismatched_pairs_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset_root(tmpdir)
            _write_pair(root, "train", "0001.jpg")
            # Missing infrared pair for 0002.jpg
            with open(
                os.path.join(root, "trainA", "0002.jpg"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("trainA:0002.jpg")

            with self.assertRaises(ValueError):
                create_dataset(
                    "VEDIA",
                    root=root,
                    split="train",
                    loader=_dummy_loader,
                )


if __name__ == "__main__":
    unittest.main()
