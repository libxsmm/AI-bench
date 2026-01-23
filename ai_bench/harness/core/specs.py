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
    RANGE = "range"
    INITS = "inits"


class InInitKey(StrEnum):
    """Keys for input initialization transforms."""

    SCALE = "scale"
    SOFTMAX = "softmax"
    ABS = "abs"
    NORMALIZE = "normalize"
    SYMMETRIC = "symmetric"
    TRI_UPPER = "triu"
    TRI_LOWER = "tril"
    TRANSPOSE = "transpose"
    UNIFORM = "uniform"
    RADEMACHER = "rademacher"


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
    HELION = "helion"


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


def input_is_float(input_entry: dict) -> bool:
    """Check if an input is of a floating point type.
    Args:
        input_entry: Specs' input entry
    Returns:
        True if type is floating point
    """
    return "float" in input_entry[InKey.TYPE]


def input_is_int(input_entry: dict) -> bool:
    """Check an input is of an integer type.
    Args:
        input_entry: Specs' input entry
    Returns:
        True if type is integer
    """
    return "int" in input_entry[InKey.TYPE]


def input_is_bool(input_entry: dict) -> bool:
    """Check an input is of a boolean type.
    Args:
        input_entry: Specs' input entry
    Returns:
        True if type is boolean
    """
    return "bool" in input_entry[InKey.TYPE]


def input_range(variant: dict, input_entry: dict) -> list[float, float]:
    """Get input's values range.
    Args:
        variant: Specs' variant entry
        input_entry: Specs' input entry
    Returns:
        Values range for the input: [low, high]
    """
    entry_range = input_entry[InKey.RANGE]
    assert len(entry_range) == 2, "Expected range with two values [low, high]"
    dims = variant[VKey.DIMS]

    def get_range_value(val: str | float) -> float:
        if isinstance(val, (int, float)):
            return val
        return dims[val]

    return [get_range_value(val) for val in entry_range]


def apply_input_inits(tensor: torch.Tensor, inits: list[str]) -> torch.Tensor:
    """
    Apply initialization transforms to a tensor.

    Transforms are applied sequentially.

    Args:
        tensor: tensor to transform
        inits: list of transform names from InInitKey
    Returns:
        transformed tensor
    """
    for init in inits:
        match InInitKey(init):
            case InInitKey.SCALE:
                scale = torch.rand((), device=tensor.device, dtype=tensor.dtype)
                tensor = tensor * scale
            case InInitKey.SOFTMAX:
                tensor = tensor.softmax(dim=-1)
            case InInitKey.ABS:
                tensor = tensor.abs()
            case InInitKey.NORMALIZE:
                tensor = torch.nn.functional.normalize(tensor, dim=-1)
            case InInitKey.SYMMETRIC:
                if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                    raise ValueError(
                        f"'{init}' requires square 2D tensor, got shape {tuple(tensor.shape)}"
                    )
                tensor = (tensor + tensor.T) / 2
            case InInitKey.TRI_UPPER:
                if tensor.ndim != 2:
                    raise ValueError(f"'{init}' requires 2D tensor, got {tensor.ndim}D")
                tensor = tensor.triu()
            case InInitKey.TRI_LOWER:
                if tensor.ndim != 2:
                    raise ValueError(f"'{init}' requires 2D tensor, got {tensor.ndim}D")
                tensor = tensor.tril()
            case InInitKey.TRANSPOSE:
                if tensor.ndim != 2:
                    raise ValueError(f"'{init}' requires 2D tensor, got {tensor.ndim}D")
                tensor = tensor.T
            case InInitKey.UNIFORM:
                # TODO: Support different bounds
                tensor = tensor.uniform_(-1, 1)
            case InInitKey.RADEMACHER:
                dist = (
                    torch.randint(0, 2, size=tensor.shape, device=tensor.device) * 2 - 1
                )
                tensor = dist.to(tensor.dtype)
    return tensor


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
        shape = input_shape(input_entry, dims)
        dtype = input_torch_dtype(input_entry)
        if variant_dtype is not None and dtype != variant_dtype:
            warnings.warn(
                f"Input '{param}' dtype ({dtype}) differs from variant dtype "
                f"({variant_dtype}). This may cause type mismatches.",
                UserWarning,
                stacklevel=2,
            )

        if input_is_float(input_entry):
            tensor = torch.randn(shape, dtype=dtype, device=device)
        elif input_is_int(input_entry):
            value_range = input_range(variant, input_entry)
            value_range = list(map(int, value_range))
            tensor = torch.randint(*value_range, shape, dtype=dtype, device=device)
        elif input_is_bool(input_entry):
            tensor = torch.randint(0, 2, shape, dtype=torch.int64, device=device).bool()
        else:
            raise TypeError("Only floating and integer types are supported now")

        if InKey.INITS in input_entry:
            tensor = apply_input_inits(tensor, input_entry[InKey.INITS])

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
