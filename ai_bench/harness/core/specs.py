from enum import StrEnum
from typing import Dict

import torch

from ai_bench import utils


class SpecKey(StrEnum):
    """Keys for spec top-level categories."""

    INS = "inputs"
    INITS = "inits"
    V_CI = "ci"
    V_BENCH_CPU = "bench-cpu"
    V_BENCH_GPU = "bench-gpu"


class InKey(StrEnum):
    """Keys for spec inputs fields."""

    SHAPE = "shape"
    TYPE = "dtype"


class InitKey(StrEnum):
    """Keys for spec inits fields."""

    DIM = "dim"


class VKey(StrEnum):
    """Keys for spec variants fields."""

    PARAMS = "params"
    DIMS = "dims"
    FLOP = "flop"


def input_shape(input: dict, dims: Dict[str, int]) -> list[int]:
    """Return shape of an input.
    Args:
        input: Specs' input entry
        dims: Specs' dimensions and their sizes
    Returns:
        List of integers defining input's shape
    """
    torch.randn
    return [dims[dim] for dim in input[InKey.SHAPE]]


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Maps specs' type to torch type.
    Args:
        dtype: Specs' data type
    Returns:
        torch data type
    """
    dtp = getattr(torch, dtype)
    return dtp


def input_torch_dtype(input: dict) -> torch.dtype:
    """Get torch data type for an input.
    Args:
        input: Specs' input entry
    Returns:
        torch data type
    """
    return get_torch_dtype(input[InKey.TYPE])


def get_inputs(
    variant: dict, inputs: dict, device: torch.device | None = None
) -> list[torch.Tensor]:
    """Get torch tensors for given specs' config.
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


def get_inits(variant: dict, inits: list[dict]) -> list[object]:
    """Get initialization values for given specs' config.
    Args:
        variant: Specs' variant entry
        inits: Specs' inits entry
    Returns:
        list of initialization values
    """
    dims = variant[VKey.DIMS]
    init_vals = []
    for init in inits:
        if InitKey.DIM in init:
            init_vals.append(dims[init[InitKey.DIM]])
        else:
            raise ValueError("Unsupported init value")
    return init_vals


def get_flop(variant: dict) -> float | None:
    """Get number of floating-point operations for given specs' variant.
    Args:
        variant: Specs' variant entry
    Returns:
        Number of FLOP if available
    """
    if VKey.FLOP not in variant:
        return None

    # Return directly if it is a number.
    flop: str | float = variant[VKey.FLOP]
    if isinstance(flop, (int, float)):
        return flop

    # In case of string equation, evaluate using variant's dimensions.
    dims = variant[VKey.DIMS]
    for dim, value in dims.items():
        flop = flop.replace(dim, str(value))
    return utils.eval_eq(flop)
