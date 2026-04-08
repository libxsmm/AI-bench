from mlir import ir
from mlir.passmanager import PassManager


def cpu_pipeline(module: ir.Module) -> ir.Module:
    """
    The default lowering pipeline for CPU.
    Lowers MLIR ops within the module to MLIR LLVM IR dialect.

    The pipeline focuses on enabling end-to-end lowering for various
    generic kernel modules.

    Performance is currently secondary and not representative.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """

    # Use standard C interface wrappers for functions.
    pm = PassManager("builtin.module", module.context)
    pm.add("func.func(llvm-request-c-wrappers)")

    # Bufferize.
    pm.add("eliminate-empty-tensors")
    pm.add(
        "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}"
    )
    pm.add("drop-equivalent-buffer-results")
    pm.add("buffer-deallocation-pipeline")
    pm.add("convert-bufferization-to-memref")
    pm.add("cse")
    pm.add("canonicalize")

    # Lower to LLVM.
    pm.add("convert-linalg-to-loops")
    pm.add("math-expand-ops")
    pm.add("expand-strided-metadata")
    pm.add("canonicalize")

    pm.add("convert-vector-to-scf")
    pm.add("lower-affine")
    pm.add("convert-scf-to-cf")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-math-to-libm")
    pm.add("convert-to-llvm")
    pm.add("reconcile-unrealized-casts")

    # Cleanup
    pm.add("cse")
    pm.add("canonicalize")

    # IR is transformed in-place.
    pm.run(module.operation)

    return module
