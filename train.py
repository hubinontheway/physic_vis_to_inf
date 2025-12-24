import argparse
import os

from datasets import create_dataset
from utils.config import load_train_config

DEFAULT_DATA_ROOT = "/data2/hubin/datasets"


def _resolve_dataset_root(config, dataset_name):
    data_root = config.get("data_root", DEFAULT_DATA_ROOT)
    dataset_root = config.get("dataset_root")
    if dataset_root:
        return dataset_root
    return os.path.join(data_root, dataset_name)


def build_dataset_from_train_config(config_path, split=None, **kwargs):
    config = load_train_config(config_path)
    dataset_name = config.get("dataset")
    if dataset_name is None:
        raise KeyError("train.yml must define 'dataset'")

    split_name = split or config.get("split", "train")
    root = _resolve_dataset_root(config, dataset_name)
    return create_dataset(dataset_name, root=root, split=split_name, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="AlignIR training entrypoint")
    parser.add_argument(
        "--config",
        default="train.yml",
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Override dataset split",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = build_dataset_from_train_config(
        args.config,
        split=args.split,
    )
    print(f"Loaded {len(dataset)} pairs")


if __name__ == "__main__":
    main()
