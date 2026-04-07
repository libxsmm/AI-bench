import lighthouse.schedule as lh_schedule
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.passmanager import PassManager


def cpu_pipeline(module: ir.Module) -> ir.Module:
    # Use standard C interface wrappers for functions.
    pm = PassManager("builtin.module", module.context)
    # pm.add("print-ir")
    pm.add("func.func(llvm-request-c-wrappers)")
    pm.run(module.operation)

    # Decompose complex Linalg ops into simpler ones.
    ctx = module.context
    with ctx, ir.Location.unknown(context=ctx):
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            # ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.conv_2d_nchw_fchw")
            # structured.structured_decompose(transform.any_op_t(), ops)
            # ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.conv_3d_ncdhw_fcdhw")
            # structured.structured_decompose(transform.any_op_t(), ops)
            softmax_ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.softmax")
            structured.structured_decompose_interface(transform.any_op_t(), softmax_ops)
            transform.yield_()
    sched.body.operations[0].apply(module)

    # Bufferize.
    pm = PassManager("builtin.module", module.context)
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
    # pm.add("print-ir")
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
    # pm.add("print-ir")

    pm.run(module.operation)

    return module
