import functools
from functools import cache
import os
from pathlib import Path
from typing import Callable

import lighthouse.dialects as lh_dialects
from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.descriptor import PipelineDescriptor
from lighthouse.pipeline.driver import BackendDriver
from mlir import ir

from ai_bench.utils import mlir_schedules_dir
from ai_bench.utils.logger import setup_logger

# Fallback pipeline descriptor, used when no target-specific pipeline is found.
_DEFAULT_PIPELINE = "default.yaml"


@cache
def _get_logger():
    return setup_logger()


@cache
def _get_target_info() -> TargetInfo:
    """
    Get target information for the current CPU architecture.

    Returns:
        TargetInfo object containing architecture and feature information.
    """
    return TargetInfo()


def _select_pipeline_file(base_path: Path, pipeline: str | None = None) -> str:
    """
    Dynamically select a lowering pipeline descriptor (YAML).

    Uses Lighthouse's ``find_pipeline_file`` to pick a target/feature/dtype
    specific pipeline from the schedules directory, falling back to the bundled
    default pipeline when no specific descriptor is available.

    Selection can be steered through environment variables:
        - AIBENCH_MLIR_SCHED_DTYPE: data type descriptor to look up (default "f32").

    Args:
        base_path: Directory holding the pipeline descriptors.
    Returns:
        Path to the selected pipeline descriptor file.
    """
    pipeline_file = str(base_path / _DEFAULT_PIPELINE)
    dtype = os.environ.get("AIBENCH_MLIR_SCHED_DTYPE", "float32")

    if pipeline:
        file, _ = PipelineDescriptor.find_pipeline_file(
            target=_get_target_info(),
            base_path=base_path,
            pipeline=pipeline,
            dtype=dtype,
        )
        if file:
            pipeline_file = file

    return pipeline_file


def cpu_pipeline(module: ir.Module, pipeline: str | None = None) -> ir.Module:
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
    pipeline_file = _select_pipeline_file(base_path, pipeline)
    _get_logger().info(f"MLIR pipeline: {pipeline_file}")

    # FIXME: Solve it in LH.
    with module.context:
        lh_dialects.register_and_load(reload=True)

    # Build the lowering pipeline from the selected YAML descriptor.
    driver = BackendDriver(module, "main", result_to_args=False, benchmark=False)
    driver.add_stage(Descriptor(pipeline_file))

    # Lower IR.
    module = driver.apply(module)

    return module


def get_cpu_compile_fn(pipeline: str | None = None) -> Callable[[ir.Module], ir.Module]:
    return functools.partial(cpu_pipeline, pipeline=pipeline)
