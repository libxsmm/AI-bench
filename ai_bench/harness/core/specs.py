from enum import StrEnum
from typing import Dict
import warnings

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
    TYPE = "dtype"
    DIMS = "dims"
    FLOP = "flop"
    MEM_BYTES = "mem_bytes"


class Backend(StrEnum):
    """Supported backends for kernel execution."""

    PYTORCH = "pytorch"
    PYTORCH_COMPILE = "pytorch-compile"
    TRITON = "triton"


def input_shape(input_entry: dict, dims: Dict[str, int]) -> list[int]:
    """Return shape of an input.
    Args:
        input_entry: Specs' input entry
        dims: Specs' dimensions and their sizes
    Returns:
        List of integers defining input's shape
    """
    return [dims[dim] for dim in input_entry[InKey.SHAPE]]


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Maps specs' type to torch type.
    Args:
        dtype: Specs' data type
    Returns:
        torch data type
    """
    dtp = getattr(torch, dtype)
    return dtp


def input_torch_dtype(input_entry: dict) -> torch.dtype:
    """Get torch data type for an input.
    Args:
        input_entry: Specs' input entry
    Returns:
        torch data type
    """
    return get_torch_dtype(input_entry[InKey.TYPE])


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
    variant_dtype = get_variant_torch_dtype(variant)
    vals = []
    for param in variant[VKey.PARAMS]:
        input_entry = inputs[param]
        assert "float" in input_entry[InKey.TYPE], "Only floating type is supported now"
        shape = input_shape(input_entry, dims)
        dtype = input_torch_dtype(input_entry)
        if variant_dtype is not None and dtype != variant_dtype:
            warnings.warn(
                f"Input '{param}' dtype ({dtype}) differs from variant dtype "
                f"({variant_dtype}). This may cause type mismatches.",
                UserWarning,
                stacklevel=2,
            )
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


def get_variant_torch_dtype(variant: dict) -> torch.dtype | None:
    """Get torch data type for given specs' variant.
    Args:
        variant: Specs' variant entry
    Returns:
        torch data type if available
    """
    if VKey.TYPE not in variant:
        return None
    return get_torch_dtype(variant[VKey.TYPE])


def _eval_variant_formula(variant: dict, key: VKey) -> float | None:
    """Evaluate a numeric or formula-based variant field.
    Args:
        variant: Specs' variant entry
        key: Specs' variant key
    Returns:
        Value if available
    """
    if key not in variant:
        return None

    # Return directly if it is a number.
    value: str | float = variant[key]
    if isinstance(value, (int, float)):
        return value

    # In case of string equation, evaluate using variant's dimensions.
    dims = variant[VKey.DIMS]
    for dim, dim_val in dims.items():
        value = value.replace(dim, str(dim_val))
    return utils.eval_eq(value)


def get_flop(variant: dict) -> float | None:
    """Get number of floating-point operations for given specs' variant.
    Args:
        variant: Specs' variant entry
    Returns:
        Number of FLOP if available
    """
    return _eval_variant_formula(variant, VKey.FLOP)


def get_mem_bytes(variant: dict) -> float | None:
    """Get number of memory access bytes for given specs' variant.
    Args:
        variant: Specs' variant entry
    Returns:
        Number of bytes if available
    """
    return _eval_variant_formula(variant, VKey.MEM_BYTES)
