import functools
from functools import cache
from pathlib import Path
from typing import Callable

from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline import find_pipeline_file
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver
from mlir import ir

from ai_bench.utils import mlir_schedules_dir
from ai_bench.utils.logger import setup_logger

# Fallback pipeline descriptor, used when no target-specific pipeline is found.
_DEFAULT_PIPELINE = "scalar-lowering.yaml"

# Maps verbose (torch-style) dtype names to the MLIR-style shorthand used in the
# pipeline descriptor filenames (e.g. ``float32`` -> ``f32``, ``bfloat16`` ->
# ``bf16``). Entries already in shorthand form pass through unchanged.
_DTYPE_ALIASES: dict[str, str] = {
    "bool": "i1",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int": "i32",
    "int64": "i64",
    "long": "i64",
    "uint8": "ui8",
    "uint16": "ui16",
    "uint32": "ui32",
    "uint64": "ui64",
}


def normalize_dtype(dtype: str) -> str:
    """
    Normalize a data type descriptor to the shorthand notation used by the
    pipeline descriptor filenames.

    Args:
        dtype: Data type descriptor (e.g. ``"float32"``, ``"bf16"``).
    Returns:
        The shorthand data type notation (e.g. ``"f32"``, ``"bf16"``).
    """
    key = dtype.strip().lower()
    return _DTYPE_ALIASES.get(key, key)


@cache
def _get_logger():
    return setup_logger()


@cache
def _get_target_info(device_type: str = "cpu") -> TargetInfo:
    """
    Get target information for the active compilation target.

    Returns:
        TargetInfo object containing architecture and feature information.
    """
    if device_type == "xpu":
        return TargetInfo(arch="xegpu", features=[])
    return TargetInfo()


def _select_pipeline_file(
    pipeline: str | None = None,
    dtype: str | None = None,
    base_path: Path | None = None,
    device_type: str = "cpu",
) -> str:
    """
    Dynamically select a lowering pipeline descriptor (YAML).

    Uses Lighthouse's ``find_pipeline_file`` to pick a target/feature/dtype
    specific pipeline from the schedules directory, falling back to the bundled
    default pipeline when no specific descriptor is available.

    Args:
        base_path: Directory holding the pipeline descriptors.
        pipeline: Optional pipeline name to select.
        dtype: Optional data type descriptor to select.
    Returns:
        Path to the selected pipeline descriptor file.
    """
    pipeline_file = _DEFAULT_PIPELINE
    if not dtype:
        dtype = "float32"

    if pipeline:
        file, _ = find_pipeline_file(
            target=_get_target_info(device_type),
            pipeline=pipeline,
            dtype=normalize_dtype(dtype),
            base_path=base_path,
        )
        if file:
            pipeline_file = file

    return pipeline_file


def _compile_pipeline(
    module: ir.Module,
    pipeline: str | None = None,
    dtype: str | None = None,
    device_type: str = "cpu",
) -> ir.Module:
    """
    The default lowering pipeline for CPU.
    Lowers MLIR ops within the module to MLIR LLVM IR dialect.

    A lowering pipeline is selected dynamically from the YAML descriptors under
    ``backends/utils/mlir_cpu_utils/schedules`` using Lighthouse's
    ``find_pipeline_file`` (based on the target architecture, feature and data
    type). When no target-specific descriptor is available it falls back to the
    bundled ``default.yaml`` pipeline, which mirrors the original hard-coded
    pass sequence.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """
    base_path = mlir_schedules_dir()
    pipeline_file = _select_pipeline_file(
        pipeline=pipeline,
        dtype=dtype,
        base_path=base_path,
        device_type=device_type,
    )
    _get_logger().info(f"  MLIR pipeline: {pipeline_file}")

    # Build the lowering pipeline from the selected YAML descriptor.
    driver = BackendDriver(module, "main", result_to_args=False, benchmark=False)
    driver.add_stage(Descriptor(pipeline_file))

    # Lower IR.
    module = driver.apply(module)

    return module


def cpu_pipeline(
    module: ir.Module, pipeline: str | None = None, dtype: str | None = None
) -> ir.Module:
    """Lower MLIR for a CPU target."""
    return _compile_pipeline(module, pipeline=pipeline, dtype=dtype, device_type="cpu")


def xpu_pipeline(
    module: ir.Module, pipeline: str | None = None, dtype: str | None = None
) -> ir.Module:
    """Lower MLIR for an XPU target."""
    return _compile_pipeline(module, pipeline=pipeline, dtype=dtype, device_type="xpu")


def get_cpu_compile_fn(
    pipeline: str | None = None, dtype: str | None = None
) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(cpu_pipeline, pipeline=pipeline, dtype=dtype)


def get_xpu_compile_fn(
    pipeline: str | None = None, dtype: str | None = None
) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(xpu_pipeline, pipeline=pipeline, dtype=dtype)
