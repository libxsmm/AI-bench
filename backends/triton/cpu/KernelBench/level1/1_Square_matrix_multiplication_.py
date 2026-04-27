# ruff: noqa: E731
# Example Triton CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}),
    ],
    key=["M", "N", "K"],  # autotune per problem size
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr, shape=(M, K), strides=(K, 1), block_shape=(BLOCK_M, BLOCK_K)
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr, shape=(K, N), strides=(N, 1), block_shape=(BLOCK_K, BLOCK_N)
    )
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_M, BLOCK_N)
    )

    m = tl.program_id(0) * BLOCK_M
    n = tl.program_id(1) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load((m, k))
        b = b_desc.load((k, n))
        acc = tl.dot(a, b, acc)

    c_desc.store((m, n), acc)


def _kernel_function_cpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    assert A.device.type == "cpu" and B.device.type == "cpu", "A and B must be on CPU"
    assert A.is_floating_point() and B.is_floating_point(), (
        "A and B must be floating point tensors"
    )
    assert A.dtype == B.dtype, f"dtype mismatch: {A.dtype} vs {B.dtype}"

    orig_dtype = A.dtype

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible K dimensions: {K} vs {K2}"

    C32 = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Autotuned grid: depends on BLOCK_M/BLOCK_N chosen by config
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_kernel[grid](
        A,
        B,
        C32,
        M,
        N,
        K,
    )

    return C32.to(orig_dtype)


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _kernel_function_cpu(A, B)
