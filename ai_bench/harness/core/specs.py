from enum import StrEnum
from typing import Dict

import torch


class InKey(StrEnum):
    """Keys for spec inputs"""

    INS = "inputs"
    SHAPE = "shape"
    TYPE = "dtype"


class VKey(StrEnum):
    """Keys for spec variants"""

    VAR_CI = "ci"
    PARAMS = "params"
    DIMS = "dims"


def input_shape(input: dict, dims: Dict[str, int]) -> list[int]:
    """Return shape of an input
    Args:
        input: Specs' input entry
        dims: Specs' dimensions and their sizes
    Returns:
        List of integers defining input's shape
    """
    torch.randn
    return [dims[dim] for dim in input[InKey.SHAPE]]


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Maps specs' type to torch type
    Args:
        dtype: Specs' data type
    Returns:
        torch data type
    """
    dtp = getattr(torch, dtype)
    return dtp


def input_torch_dtype(input: dict) -> torch.dtype:
    """Get torch data type for an input
    Args:
        input: Specs' input entry
    Returns:
        torch data type
    """
    return get_torch_dtype(input[InKey.TYPE])


def get_inputs(
    variant: dict, inputs: dict, device: torch.device | None = None
) -> list[torch.Tensor]:
    """Get torch tensors for given specs' config
    Args:
        variant: Specs' variant entry
        inputs: Specs' inputs entry
        device: Desired device of the tensors
    Returns:
        list of torch tensors
    """
    dims = variant[VKey.DIMS]
    vals = []
    for param in variant[VKey.PARAMS]:
        input = inputs[param]
        assert "float" in input[InKey.TYPE], "Only floating type is supported now"
        shape = input_shape(input, dims)
        dtype = input_torch_dtype(input)
        tensor = torch.randn(shape, dtype=dtype, device=device)
        vals.append(tensor)
    return vals
