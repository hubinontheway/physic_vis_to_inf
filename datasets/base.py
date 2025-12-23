import os


def default_image_loader(path, mode=None):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image loading. "
            "Install it or pass a custom loader."
        ) from exc

    with Image.open(path) as img:
        if mode:
            img = img.convert(mode)
        return img


class PairedImageDataset:
    """Base dataset for paired visible/infrared images."""

    def __init__(
        self,
        root,
        split="train",
        loader=None,
        transform=None,
        vis_mode="RGB",
        ir_mode="L",
    ):
        self.root = root
        self.split = split.lower()
        self.loader = loader or default_image_loader
        self.transform = transform
        self.vis_mode = vis_mode
        self.ir_mode = ir_mode

        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.vis_dir = os.path.join(self.root, f"{self.split}A")
        self.ir_dir = os.path.join(self.root, f"{self.split}B")

        if not os.path.isdir(self.vis_dir):
            raise FileNotFoundError(f"Visible folder not found: {self.vis_dir}")
        if not os.path.isdir(self.ir_dir):
            raise FileNotFoundError(f"Infrared folder not found: {self.ir_dir}")

        self.pairs = self._build_pairs()

    def _list_files(self, directory):
        return sorted(
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        )

    def _build_pairs(self):
        vis_files = self._list_files(self.vis_dir)
        ir_files = self._list_files(self.ir_dir)
        vis_set = set(vis_files)
        ir_set = set(ir_files)

        if vis_set != ir_set:
            missing_vis = sorted(ir_set - vis_set)
            missing_ir = sorted(vis_set - ir_set)
            raise ValueError(
                "Mismatched pairs: "
                f"missing in A={missing_vis}, missing in B={missing_ir}"
            )

        return [
            (os.path.join(self.vis_dir, name), os.path.join(self.ir_dir, name))
            for name in vis_files
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        vis_path, ir_path = self.pairs[index]
        vis = self.loader(vis_path, mode=self.vis_mode)
        ir = self.loader(ir_path, mode=self.ir_mode)

        if self.transform is not None:
            transformed = self.transform(vis, ir)
            if (
                not isinstance(transformed, tuple)
                or len(transformed) != 2
            ):
                raise ValueError("transform must return (vis, ir)")
            vis, ir = transformed

        return {
            "vis": vis,
            "ir": ir,
            "vis_path": vis_path,
            "ir_path": ir_path,
        }
