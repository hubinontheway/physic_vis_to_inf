from .paired import AVIIDDataset, M3FDDataset, VEDIADataset


DATASET_REGISTRY = {
    VEDIADataset.name: VEDIADataset,
    AVIIDDataset.name: AVIIDDataset,
    M3FDDataset.name: M3FDDataset,
}


def get_dataset_class(name):
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]


def create_dataset(name, root, split="train", **kwargs):
    dataset_class = get_dataset_class(name)
    return dataset_class(root=root, split=split, **kwargs)
