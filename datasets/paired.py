from .base import PairedImageDataset


class VEDIADataset(PairedImageDataset):
    name = "VEDIA"


class AVIIDDataset(PairedImageDataset):
    name = "AVIID"


class M3FDDataset(PairedImageDataset):
    name = "M3FD_Detection"
