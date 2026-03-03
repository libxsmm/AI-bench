# Example Helion XPU kernel
# Source: helion matmul example
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import helion
import helion.language as hl
import torch
import torch.nn as nn


@helion.kernel(
    static_shapes=True,
    configs=[
        helion.Config(
            block_sizes=[64, 128, 16],
            indexing="tensor_descriptor",
            l2_groupings=[32],
            loop_orders=[[1, 0]],
            num_stages=2,
            num_warps=8,
            pid_type="flat",
            range_flattens=[None, None],
            range_multi_buffers=[None, None],
            range_num_stages=[0, 2],
            range_unroll_factors=[0, 1],
        ),
        helion.Config(
            block_sizes=[256, 256, 32],
            indexing="tensor_descriptor",
            l2_groupings=[4],
            loop_orders=[[0, 1]],
            num_stages=2,
            num_warps=32,
            pid_type="flat",
            range_flattens=[None, False],
            range_multi_buffers=[None, False],
            range_num_stages=[0, 2],
            range_unroll_factors=[0, 1],
        ),
        helion.Config(
            block_sizes=[256, 128, 32],
            indexing="tensor_descriptor",
            l2_groupings=[32],
            loop_orders=[[0, 1]],
            num_stages=4,
            num_warps=32,
            pid_type="persistent_interleaved",
            range_flattens=[None, False],
            range_multi_buffers=[True, False],
            range_num_stages=[1, 4],
            range_unroll_factors=[4, 1],
        ),
        helion.Config(
            block_sizes=[128, 256, 16],
            indexing="tensor_descriptor",
            l2_groupings=[4],
            loop_orders=[[0, 1]],
            num_stages=5,
            num_warps=32,
            pid_type="persistent_interleaved",
            range_flattens=[None, True],
            range_multi_buffers=[False, False],
            range_num_stages=[1, 4],
            range_unroll_factors=[2, 0],
        ),
    ],
)
def _square_matmul_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs square matrix multiplication using Helion.
    C = A * B

    Args:
        A: Input matrix A of shape (N, N)
        B: Input matrix B of shape (N, N)

    Returns:
        Output matrix C of shape (N, N)
    """
    N, N2 = A.size()
    N3, N4 = B.size()
    assert N == N2 == N3 == N4, f"size mismatch: A{A.size()}, B{B.size()}"

    out = torch.empty(
        [N, N], dtype=torch.promote_types(A.dtype, B.dtype), device=A.device
    )

    for tile_m, tile_n in hl.tile([N, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(N):
            acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _square_matmul_kernel(A, B)
