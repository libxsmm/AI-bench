# ruff: noqa: E731
# Example Triton CUDA kernel
# Source: triton-lang/triton matmul tutorial
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4
        ),
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
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _kernel_function_cuda(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    assert hasattr(torch, "cuda"), "torch.cuda is required for this kernel"
    assert A.device.type == "cuda" and B.device.type == "cuda", (
        "A and B must be on CUDA"
    )
    assert A.is_floating_point() and B.is_floating_point(), (
        "A and B must be floating point tensors"
    )
    assert A.dtype == B.dtype, f"dtype mismatch: {A.dtype} vs {B.dtype}"

    orig_dtype = A.dtype

    K, M = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible K dimensions: {K} vs {K2}"

    C32 = torch.empty((M, N), device=A.device, dtype=torch.float32)

    stride_A0, stride_A1 = A.stride()
    stride_B0, stride_B1 = B.stride()
    stride_C0, stride_C1 = C32.stride()

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
        stride_A0,
        stride_A1,
        stride_B0,
        stride_B1,
        stride_C0,
        stride_C1,
    )

    torch.accelerator.synchronize()
    return C32.to(orig_dtype)


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _kernel_function_cuda(A, B)
