from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import gpu
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector
from mlir.passmanager import PassManager
import torch
import torch.nn as nn

import ai_bench.mlir

TILE_SIZE = 64


def tile_and_vector_gemm(ctx: ir.Context) -> ir.Module:
    """
    Specialized schedule for Linalg operations.

    Tiling and vectorization is progressively applied to
    achieve SIMD code generation.

    Args:
        ctx: MLIR context.
    Returns:
        MLIR transform module.
    """
    with ctx, ir.Location.unknown(context=ctx):
        # Create a transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.any_op_t()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            anytype = transform.any_op_t()

            # GEMM tiling.
            mm = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.matmul"]
            ).result
            tiled_mm = structured.FuseOp(
                mm, tile_sizes=[TILE_SIZE, TILE_SIZE], apply_cleanup=True
            ).results[0]

            # Tile buffer initialization for better vectorization.
            tiled_fill = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.fill"]
            ).result
            reg_fill = structured.TileUsingForOp(
                tiled_fill, sizes=[1, TILE_SIZE]
            ).results[0]

            # Register tiling.
            reg_mm = structured.TileUsingForOp(tiled_mm, sizes=[8, 32, 1]).results[0]

            # Vectorize operations.
            structured.structured_vectorize(reg_mm, [], create_named_contraction=True)
            structured.structured_vectorize(reg_fill, [])

            # Loop hoisting.
            all_loops = structured.MatchOp(
                anytype,
                named_seq.bodyTarget,
                interface=structured.MatchInterfaceEnum.LoopLikeInterface,
            ).results
            transform.apply_licm(all_loops)
            loop.loop_hoist_loop_invariant_subsets(all_loops)

            # Unroll GEMM.
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                gpu.apply_patterns_gpu_unroll_vectors_subgroup_mma(m=1, n=32, k=1)
                vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
                transform.apply_patterns_canonicalization()

            # Lower to broadcast+FMA instructions.
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                vector.apply_patterns_vector_lower_contraction(
                    lowering_strategy=vector.VectorContractLowering.OuterProduct
                )
                vector.apply_patterns_vector_lower_outerproduct()

            # Cleanup.
            transform.apply_cse(named_seq.bodyTarget)
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                transform.apply_patterns_canonicalization()

            transform.yield_()
    return schedule


def lower_to_llvm(module: ir.Module) -> ir.Module:
    """
    Lower MLIR ops within the module to MLIR LLVM IR dialect.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """
    pm = PassManager("builtin.module", module.context)

    # Preprocess.
    # Use standard C interface wrappers for functions.
    pm.add("func.func(llvm-request-c-wrappers)")

    # Apply schedule.
    sched = tile_and_vector_gemm(module.context)
    sched.body.operations[0].apply(module)

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
    pm.add("expand-strided-metadata")
    pm.add("canonicalize")

    pm.add("convert-vector-to-scf")
    pm.add("lower-affine")
    pm.add("convert-scf-to-cf")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-to-llvm")
    pm.add("reconcile-unrealized-casts")

    # Cleanup
    pm.add("cse")
    pm.add("canonicalize")

    # IR is transformed in-place.
    pm.run(module.operation)

    # Return the same module which now holds LLVM IR dialect ops.
    return module


@torch.compile(dynamic=False, backend=ai_bench.mlir.cpu_backend(lower_to_llvm))
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert all(dim % TILE_SIZE == 0 for dim in A.shape), (
            f"A shape must be multiple of {TILE_SIZE}"
        )
        assert all(dim % TILE_SIZE == 0 for dim in B.shape), (
            f"B shape must be multiple of {TILE_SIZE}"
        )

        return torch.matmul(A, B)
