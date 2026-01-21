import torch


def apply_scale(tensor: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    if scale is None:
        scale = torch.rand(())
    return tensor * scale


def apply_softmax(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return tensor.softmax(dim)
