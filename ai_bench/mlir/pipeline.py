import functools
import os
import sys
from functools import cache
from pathlib import Path
from typing import Callable

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline import find_pipeline_file
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver
from lighthouse.schedule.xegpu import (
    XeGPUParameterSelector,
    elemwise_schedule,
    mlp_schedule,
    xegpu_to_binary,
)
from lighthouse.utils.mlir import inspect_payload
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
    logger = _get_logger()
    logger.info(f"BackendDriver: creating driver for module with context {id(module.context)}")
    driver = BackendDriver(module, "main", result_to_args=False, benchmark=False)
    logger.info(f"BackendDriver: adding stage from {pipeline_file}")
    driver.add_stage(Descriptor(pipeline_file))
    logger.info(f"BackendDriver: applying pipeline...")

    # Lower IR.
    # In debug mode, apply stages one by one so we can identify the exact stage
    # that aborts when MLIR raises a fatal C++ assertion.
    try:
        if os.environ.get("AIBENCH_MLIR_TRACE_STAGES"):
            for i, stage in enumerate(driver.stages):
                stage_module = getattr(stage, "module", None)
                stage_schedule = getattr(stage, "schedule", None)
                if stage_module is not None:
                    has_named_seq_attr = (
                        "transform.with_named_sequence"
                        in stage_module.operation.attributes
                    )
                    first_op_name = (
                        stage_module.body.operations[0].name
                        if len(stage_module.body.operations) > 0
                        else "<empty>"
                    )
                else:
                    has_named_seq_attr = None
                    first_op_name = "<n/a>"
                schedule_op_name = (
                    stage_schedule.name if stage_schedule is not None else "<n/a>"
                )
                print(
                    "AIBENCH_TRACE: "
                    f"applying stage[{i}] "
                    f"schedule_op={schedule_op_name} "
                    f"module_has_named_seq={has_named_seq_attr} "
                    f"module_first_op={first_op_name} "
                    f"stage={stage}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.info(f"BackendDriver: applying stage[{i}] {stage}")
                with module.context:
                    module = stage.apply(module)
                print(
                    f"AIBENCH_TRACE: stage[{i}] applied",
                    file=sys.stderr,
                    flush=True,
                )
                logger.info(f"BackendDriver: stage[{i}] applied")
            logger.info("BackendDriver: pipeline applied successfully")
        else:
            module = driver.apply(module)
            logger.info(f"BackendDriver: pipeline applied successfully")
    except Exception as e:
        logger.error(f"BackendDriver.apply() failed: {type(e).__name__}: {e}")
        logger.error(f"Module context id: {id(module.context)}")
        raise

    return module


def _infer_xpu_kernel_metadata(
    module: ir.Module,
    payload_func_name: str = "main",
) -> tuple[str, list[dict]]:
    """Infer schedule kind and params from payload metadata.

    Returns:
        Tuple of (schedule_kind, schedule_params)
        schedule_kind in {"mlp", "elemwise", "unsupported"}
    """
    payload = inspect_payload(module)
    func_meta = payload.get(payload_func_name)
    if not func_meta:
        return "unsupported", []

    layers = func_meta.get("layers", {})
    matmuls = layers.get("matmul", [])
    if matmuls:
        selector = XeGPUParameterSelector()
        params = selector.get_parameters_for_layers(matmuls)
        return "mlp", params

    elemwise_layers = layers.get("elemwise", [])
    if elemwise_layers:
        return "elemwise", elemwise_layers

    return "unsupported", []


def _compile_xpu_adaptive(
    module: ir.Module,
    payload_func_name: str = "main",
) -> ir.Module:
    """Lower XPU payload using adaptive Lighthouse schedules."""
    logger = _get_logger()
    schedule_kind, schedule_params = _infer_xpu_kernel_metadata(
        module, payload_func_name=payload_func_name
    )

    if schedule_kind == "unsupported":
        raise ValueError("Unsupported XPU payload for adaptive schedule")

    with module.context, ir.Location.unknown():
        # Ensure custom Lighthouse transform dialect extensions are loaded in
        # this payload context before constructing schedules.
        lh_dialects.register_and_load()

        if schedule_kind == "mlp":
            schedule = mlp_schedule(
                params=schedule_params,
                payload_func_name=payload_func_name,
            )
        else:
            schedule = elemwise_schedule(
                params=schedule_params,
                payload_func_name=payload_func_name,
            )
        lower_to_binary = xegpu_to_binary(
            xegpu_op_level="workgroup",
            large_register_file=True,
        )

    driver = BackendDriver(module, payload_func_name, result_to_args=False, benchmark=False)
    driver.add_transform(schedule)
    driver.add_transform(lower_to_binary)

    if os.environ.get("AIBENCH_MLIR_TRACE_STAGES"):
        logger.info(
            f"XPU adaptive schedule kind={schedule_kind}, stages={len(driver.stages)}"
        )
        for i, stage in enumerate(driver.stages):
            stage_module = getattr(stage, "module", None)
            stage_schedule = getattr(stage, "schedule", None)
            if stage_module is not None:
                has_named_seq_attr = (
                    "transform.with_named_sequence"
                    in stage_module.operation.attributes
                )
                first_op_name = (
                    stage_module.body.operations[0].name
                    if len(stage_module.body.operations) > 0
                    else "<empty>"
                )
            else:
                has_named_seq_attr = None
                first_op_name = "<n/a>"
            schedule_op_name = (
                stage_schedule.name if stage_schedule is not None else "<n/a>"
            )
            print(
                "AIBENCH_TRACE_ADAPTIVE: "
                f"applying stage[{i}] "
                f"schedule_op={schedule_op_name} "
                f"module_has_named_seq={has_named_seq_attr} "
                f"module_first_op={first_op_name} "
                f"stage={stage}",
                file=sys.stderr,
                flush=True,
            )
            with module.context:
                module = stage.apply(module)
            print(
                f"AIBENCH_TRACE_ADAPTIVE: stage[{i}] applied",
                file=sys.stderr,
                flush=True,
            )
        return module

    return driver.apply(module)


def cpu_pipeline(
    module: ir.Module, pipeline: str | None = None, dtype: str | None = None
) -> ir.Module:
    """Lower MLIR for a CPU target."""
    return _compile_pipeline(module, pipeline=pipeline, dtype=dtype, device_type="cpu")


def xpu_pipeline(
    module: ir.Module, pipeline: str | None = None, dtype: str | None = None
) -> ir.Module:
    """Lower MLIR for an XPU target."""
    logger = _get_logger()
    try:
        logger.info("XPU pipeline: using adaptive schedule construction")
        return _compile_xpu_adaptive(module, payload_func_name="main")
    except Exception as e:
        logger.warning(
            "XPU adaptive scheduling failed; falling back to descriptor pipeline: "
            f"{type(e).__name__}: {e}"
        )
        return _compile_pipeline(
            module,
            pipeline=pipeline,
            dtype=dtype,
            device_type="xpu",
        )


def get_cpu_compile_fn(
    pipeline: str | None = None, dtype: str | None = None
) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(cpu_pipeline, pipeline=pipeline, dtype=dtype)


def get_xpu_compile_fn(
    pipeline: str | None = None, dtype: str | None = None
) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(xpu_pipeline, pipeline=pipeline, dtype=dtype)
