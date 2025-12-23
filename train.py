import argparse

from datasets import create_dataset
from utils.config import load_dataset_paths


def build_dataset_from_config(config_path, dataset_name, split="train", **kwargs):
    dataset_paths = load_dataset_paths(config_path)
    if dataset_name not in dataset_paths:
        available = ", ".join(sorted(dataset_paths))
        raise KeyError(
            f"Dataset '{dataset_name}' not found in config. "
            f"Available: {available}"
        )

    root = dataset_paths[dataset_name]
    return create_dataset(dataset_name, root=root, split=split, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="AlignIR training entrypoint")
    parser.add_argument(
        "--config",
        default="configs/datasets.yml",
        help="Path to dataset YAML config",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["VEDIA", "AVIID", "M3FD_Detection"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = build_dataset_from_config(
        args.config,
        dataset_name=args.dataset,
        split=args.split,
    )
    print(f"Loaded {args.dataset} ({args.split}) with {len(dataset)} pairs")


if __name__ == "__main__":
    main()
