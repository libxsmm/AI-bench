from collections.abc import Callable
from collections.abc import Sequence
import os
from pathlib import Path
import warnings

import lighthouse.ingress.torch.compile as lh_compile
from mlir import ir
import torch

from ai_bench.utils.logger import setup_logger

_XPU_SHARED_IR_CONTEXT: ir.Context | None = None


def _get_xpu_shared_ir_context() -> ir.Context:
    """Return a process-wide MLIR context for XPU compilation."""
    global _XPU_SHARED_IR_CONTEXT
    if _XPU_SHARED_IR_CONTEXT is None:
        _XPU_SHARED_IR_CONTEXT = ir.Context()
    return _XPU_SHARED_IR_CONTEXT


# TODO: Add proper discovery to 'finder' module if params need to be kept locally.
def _xpu_matmul_params_file() -> str:
    """Return the local XeGPU matmul parameter database path."""
    return str(
        Path(__file__).resolve().parents[2]
        / "backends"
        / "utils"
        / "mlir_xpu_utils"
        / "matmul_params.json"
    )


# TODO: Fix configuration in Lighthouse.
def _override_lighthouse_xpu_param_db(logger) -> None:
    """Point Lighthouse XeGPU parameter selector to AI-bench's local JSON DB."""
    from lighthouse.schedule.xegpu import xegpu_parameter_selector

    json_file = _xpu_matmul_params_file()
    if not Path(json_file).exists():
        raise FileNotFoundError(f"Missing XPU matmul params file: {json_file}")
    xegpu_parameter_selector.DEFAULT_JSON_FILE = json_file
    logger.info(f"--- MLIR XPU - Using matmul params: {json_file}")


class CPUBackend(lh_compile.MLIRBackend):
    """
    A wrapper around PyTorch MLIR CPU backend.
    Overrides to inject extras through environment variables.

    Args:
        device: Target device.
        fn_compile: Function to lower imported MLIR to LLVM IR dialect.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
            If not provided, a new default context is created.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function. Extra paths provided through environment
            variable are also included.
    """

    def __init__(
        self,
        device: torch.device,
        fn_compile: Callable[[ir.Module], ir.Module],
        dialect: lh_compile.TargetDialect = lh_compile.TargetDialect.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
        **kwargs,
    ):
        self.logger = setup_logger()

        shared_libs = list(shared_libs)
        lib_paths = os.environ.get("AIBENCH_MLIR_LIB_PATH")
        if lib_paths:
            libs = lib_paths.split(":")
            shared_libs.extend(libs)
        super().__init__(
            device,
            fn_compile,
            dialect,
            ir_context,
            shared_libs=shared_libs,
            entry_func="main",
            **kwargs,
        )
        assert self.entry_func == "main", "Expected entry function to be 'main'"

    def get_mlir(
        self, model: torch.nn.Module, example_inputs: list[torch.Tensor]
    ) -> ir.Module:
        """
        Convert PyTorch model to MLIR IR.
        Overrides to inject debug info.

        Args:
            model: PyTorch model.
            example_inputs: Inputs to the model.
        Returns:
            MLIR module.
        """
        mlir_mod = super().get_mlir(model, example_inputs)

        if os.environ.get("AIBENCH_MLIR_DUMP"):
            self.logger.info("--- MLIR JIT - Imported IR:\n" + str(mlir_mod))

        return mlir_mod

    def __call__(
        self, model: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        """
        Import a PyTorch model into MLIR and return a compiled function.
        Overrides to inject debug info.

        Args:
            model: Traced PyTorch model.
            example_inputs: Example input tensors.

        Returns:
            Callable function.
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
        jit_func = super().__call__(model, example_inputs)

        if os.environ.get("AIBENCH_MLIR_DUMP_OBJ"):
            import uuid

            file = "jit-mlir-dump-" + uuid.uuid4().hex + ".o"
            jit_func.eng.dump_to_object_file(file)
            self.logger.info(f"--- MLIR JIT - Created object file: {file}")

        return jit_func


class GPUBackend(CPUBackend):
    """
    A wrapper around PyTorch MLIR GPU backend.

    Reuses the AI-bench MLIR wrapper behavior while targeting an accelerator
    device supported by Lighthouse's GPU backend.
    """

    def __init__(
        self,
        device: torch.device,
        fn_compile: Callable[[ir.Module], ir.Module],
        dialect: lh_compile.TargetDialect = lh_compile.TargetDialect.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
        **kwargs,
    ):
        assert device.type in ("cuda", "rocm", "xpu"), "Expected a GPU device"
        logger = setup_logger()
        if device.type == "xpu":
            _override_lighthouse_xpu_param_db(logger)
            shared_libs = list(shared_libs)
            shared_libs.append("libmlir_levelzero_runtime.so")
            if ir_context is None:
                ir_context = _get_xpu_shared_ir_context()
        super().__init__(
            device,
            fn_compile,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
            **kwargs,
        )


def cpu_backend(
    fn_compile: Callable[[ir.Module], ir.Module],
    dialect: lh_compile.TargetDialect = lh_compile.TargetDialect.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
    **kwargs,
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable]:
    """
    CPU backend for JIT-compiling a PyTorch model using MLIR.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module, and returns an MLIR module with
            transformed IR.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.

    Returns:
        object: A PyTorch model or a partially bound decorator.
    """
    return CPUBackend(
        torch.device("cpu"),
        fn_compile,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
        **kwargs,
    )


def gpu_backend(
    fn_compile: Callable[[ir.Module], ir.Module],
    device: torch.device,
    dialect: lh_compile.TargetDialect = lh_compile.TargetDialect.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
    **kwargs,
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable]:
    """
    GPU backend for JIT-compiling a PyTorch model using MLIR.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module, and returns an MLIR module with
            transformed IR.
        device: Target GPU device.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.

    Returns:
        object: A PyTorch model or a partially bound decorator.
    """
    return GPUBackend(
        device,
        fn_compile,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
        **kwargs,
    )
