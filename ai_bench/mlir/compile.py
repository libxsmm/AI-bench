from collections.abc import Callable
from collections.abc import Sequence
import os

import lighthouse.ingress.torch.compile as lh_compile
from mlir import ir
import torch

from ai_bench.utils.logger import setup_logger


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
            device, fn_compile, dialect, ir_context, shared_libs=shared_libs, **kwargs
        )

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
        jit_func = super().__call__(model, example_inputs)

        if os.environ.get("AIBENCH_MLIR_DUMP_OBJ"):
            import uuid

            file = "jit-mlir-dump-" + uuid.uuid4().hex + ".o"
            jit_func.eng.dump_to_object_file(file)
            self.logger.info(f"--- MLIR JIT - Created object file: {file}")

        return jit_func


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
