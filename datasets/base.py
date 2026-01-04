import os


from torch.utils.data import Dataset


def default_image_loader(path, mode=None):
    """
    Default image loader using PIL.
    
    This function loads an image from the specified path using PIL and optionally
    converts it to the specified mode (e.g., 'RGB', 'L').
    
    Args:
        path (str): Path to the image file
        mode (str, optional): PIL mode to convert the image to (e.g., 'RGB', 'L')
        
    Returns:
        PIL.Image: Loaded image object
        
    Raises:
        ImportError: If PIL is not installed
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image loading. "
            "Install it or pass a custom loader."
        ) from exc

    with Image.open(path) as img:
        if mode:
            # Convert image to specified mode if provided
            img = img.convert(mode)
        return img


class PairedImageDataset(Dataset):
    """
    Base dataset for paired visible/infrared images.
    
    This dataset class handles the loading and processing of paired visible and
    infrared images. It assumes that the dataset is organized with two folders:
    one for visible images (split + 'A', e.g., 'trainA') and one for infrared
    images (split + 'B', e.g., 'trainB'), where corresponding images have the
    same filename.
    
    The class validates that all files have corresponding pairs and provides
    a standardized interface for accessing the paired data.
    """

    def __init__(
        self,
        root,
        split="train",
        loader=None,
        transform=None,
        vis_mode="RGB",
        ir_mode="L",
    ):
        """
        Initialize the PairedImageDataset.
        
        Args:
            root (str): Root directory of the dataset
            split (str): Dataset split, either 'train' or 'test'. Defaults to 'train'.
            loader (callable, optional): Function to load images. If None, uses default loader.
            transform (callable, optional): Function to transform image pairs.
            vis_mode (str): PIL mode for visible images. Defaults to 'RGB'.
            ir_mode (str): PIL mode for infrared images. Defaults to 'L' (grayscale).
        
        Raises:
            ValueError: If split is not 'train' or 'test'
            FileNotFoundError: If visible or infrared directories don't exist
        """
        self.root = root
        self.split = split.lower()
        self.loader = loader or default_image_loader  # Use provided loader or default
        self.transform = transform  # Optional transformation function
        self.vis_mode = vis_mode  # Image mode for visible images
        self.ir_mode = ir_mode  # Image mode for infrared images

        # Validate split parameter
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        # Define paths for visible and infrared image directories
        self.vis_dir = os.path.join(self.root, f"{self.split}A")
        self.ir_dir = os.path.join(self.root, f"{self.split}B")

        # Validate that both directories exist
        if not os.path.isdir(self.vis_dir):
            raise FileNotFoundError(f"Visible folder not found: {self.vis_dir}")
        if not os.path.isdir(self.ir_dir):
            raise FileNotFoundError(f"Infrared folder not found: {self.ir_dir}")

        # Build list of paired image file paths
        self.pairs = self._build_pairs()

    def _list_files(self, directory):
        """
        List all files in the specified directory.
        
        Args:
            directory (str): Path to the directory
            
        Returns:
            List[str]: Sorted list of filenames in the directory
        """
        return sorted(
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        )

    def _build_pairs(self):
        """
        Build list of paired image file paths.
        
        This method ensures that for every visible image there's a corresponding
        infrared image with the same filename, and vice versa. If there's a
        mismatch, it raises an error with details about missing files.
        
        Returns:
            List[Tuple[str, str]]: List of (visible_path, infrared_path) tuples
            
        Raises:
            ValueError: If there are mismatched pairs between visible and infrared images
        """
        vis_files = self._list_files(self.vis_dir)
        ir_files = self._list_files(self.ir_dir)
        vis_set = set(vis_files)
        ir_set = set(ir_files)

        # Check for mismatched files and raise an error if found
        if vis_set != ir_set:
            missing_vis = sorted(ir_set - vis_set)  # Files in IR but not in VIS
            missing_ir = sorted(vis_set - ir_set)  # Files in VIS but not in IR
            raise ValueError(
                "Mismatched pairs: "
                f"missing in A={missing_vis}, missing in B={missing_ir}"
            )

        # Create pairs of file paths
        return [
            (os.path.join(self.vis_dir, name), os.path.join(self.ir_dir, name))
            for name in vis_files
        ]

    def __len__(self):
        """
        Return the number of image pairs in the dataset.
        
        Returns:
            int: Number of paired images
        """
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Get a paired image at the specified index.
        
        Args:
            index (int): Index of the image pair to retrieve
            
        Returns:
            dict: Dictionary containing:
                - 'vis': Loaded visible image
                - 'ir': Loaded infrared image
                - 'vis_path': Path to visible image file
                - 'ir_path': Path to infrared image file
        """
        # Get the paths for visible and infrared images at this index
        vis_path, ir_path = self.pairs[index]
        
        # Load the images using the specified loader and modes
        vis = self.loader(vis_path, mode=self.vis_mode)
        ir = self.loader(ir_path, mode=self.ir_mode)

        # Apply transformation if provided
        if self.transform is not None:
            transformed = self.transform(vis, ir)
            # Validate that transformation returns a tuple of two elements
            if (
                not isinstance(transformed, tuple)
                or len(transformed) != 2
            ):
                raise ValueError("transform must return (vis, ir)")
            vis, ir = transformed

        # Return a dictionary with the loaded images and their paths
        return {
            "vis": vis,
            "ir": ir,
            "vis_path": vis_path,
            "ir_path": ir_path,
        }
