import functools
from functools import cache
from pathlib import Path
from typing import Callable

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline import find_pipeline_file
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver
from lighthouse.schedule.xegpu import elemwise_schedule
from lighthouse.schedule.xegpu import (
    mlp_schedule,
    elemwise_schedule,
    reduction_schedule,
)
from lighthouse.schedule.xegpu import xegpu_to_binary
from lighthouse.schedule.parameters import ScheduleParameters
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

    with module.context, ir.Location.unknown():
        # Build the lowering pipeline from the selected YAML descriptor.
        driver = BackendDriver(module, "main", result_to_args=False, benchmark=False)
        driver.add_stage(Descriptor(pipeline_file))

        # Lower IR.
        return driver.apply(module)


@cache
def _xpu_matmul_params_dir() -> str:
    """Return the local XeGPU matmul parameter database directory."""
    return str(
        Path(__file__).resolve().parents[2]
        / "backends"
        / "utils"
        / "mlir_xpu_utils"
    )


def _compile_xpu_pipeline(
    module: ir.Module,
    pipeline: str | None = None,
    parameters: ScheduleParameters | None = None,
    dtype: str | None = None,
    device_type: str = "cpu",
) -> ir.Module:
    """
    Construct lowering pipeline for XPU target using a specific schedule kind and parameters.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """

    with module.context, ir.Location.unknown():
        # High-level problem specific schedule.
        payload_func_name = "main"
        if pipeline == "mlp":
            schedule = mlp_schedule(
                params=parameters,
                payload_func_name=payload_func_name,
                device="B70",
            )
        elif pipeline == "elemwise":
            schedule = elemwise_schedule(
                params=parameters,
                payload_func_name=payload_func_name,
            )
        elif pipeline == "reduction":
            schedule = reduction_schedule(
                params=parameters,
                payload_func_name=payload_func_name,
            )
        else:
            raise ValueError(f"Unsupported XPU schedule kind: {pipeline}")
        # Full pipeline with a common tail.
        schedules = [
            schedule,
            xegpu_to_binary(
                xegpu_op_level="workgroup",
            ),
        ]
        # Build the lowering pipeline from the selected YAML descriptor.
        driver = BackendDriver(module, "main", result_to_args=False, benchmark=False)
        for s in schedules:
            driver.add_transform(s)

        # Lower IR.
        return driver.apply(module)


def _clone_module(module: ir.Module) -> ir.Module:
    """Return a deep copy of an MLIR module across binding versions."""
    with module.context, ir.Location.unknown():
        # Newer bindings may expose Module.clone().
        clone_fn = getattr(module, "clone", None)
        if callable(clone_fn):
            return clone_fn()

        # Some builds expose clone on the top-level operation.
        op_clone_fn = getattr(module.operation, "clone", None)
        if callable(op_clone_fn):
            # In some bindings, ir.Module has no public constructor.
            return ir.Module.parse(str(op_clone_fn()))

        # Portable fallback: round-trip through assembly in same context.
        return ir.Module.parse(str(module))


def cpu_pipeline(
    module: ir.Module, pipeline: str | None = None, dtype: str | None = None
) -> ir.Module:
    """Lower MLIR for a CPU target."""
    return _compile_pipeline(module, pipeline=pipeline, dtype=dtype, device_type="cpu")


def xpu_pipeline(
    module: ir.Module, pipeline: str | None = None, pipeline_parameters: str | None = None, dtype: str | None = None
) -> ir.Module:
    """Lower MLIR for an XPU target."""
    logger = _get_logger()
    if not pipeline or not pipeline_parameters:
        logger.debug("XPU lowering requires both pipeline and pipeline_parameters.")
        raise ValueError("XPU lowering requires pipeline and pipeline_parameters.")
    try:
        # Prepend parameters file with local parameter database directory.
        pipeline_parameters = str(Path(_xpu_matmul_params_dir()).joinpath(pipeline_parameters))
        logger.info(f"  MLIR XPU - schedule kind: {pipeline}")
        logger.info(f"  MLIR XPU - loading schedule parameters from: {pipeline_parameters}")
        schedule_parameters = ScheduleParameters.from_json(pipeline_parameters)
        # Clone module to avoid mutating the original in case of failure.
        payload = _clone_module(module)
        return _compile_xpu_pipeline(
            payload,
            pipeline=pipeline,
            parameters=schedule_parameters,
            dtype=dtype,
            device_type="xpu",
        )
    except Exception as e:
        logger.debug("  XPU default schedule failed; using fallback...")
        logger.debug(f"  XPU compile error:\n{e}")


def get_cpu_compile_fn(
    pipeline: str | None = None, dtype: str | None = None
) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(cpu_pipeline, pipeline=pipeline, dtype=dtype)


def get_xpu_compile_fn(
    pipeline: str | None = None, pipeline_parameters: str | None = None, dtype: str | None = None
) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(xpu_pipeline, pipeline=pipeline, pipeline_parameters=pipeline_parameters, dtype=dtype)
