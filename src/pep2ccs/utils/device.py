"""Select CPU or the default CUDA device. Never pin a physical GPU index."""

import torch


def get_device(requested: str = None) -> torch.device:
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device
