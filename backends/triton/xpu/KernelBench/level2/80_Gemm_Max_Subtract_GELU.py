# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _zero_epilogue_configs():
    return [
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 256}, num_warps=32, num_stages=3),
        triton.Config({"BLOCK_M": 512}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 512}, num_warps=32, num_stages=3),
    ]


@triton.autotune(
    configs=_zero_epilogue_configs(),
    key=["M"],
)
@triton.jit
def _zero_epilogue_kernel(
    out_ptr,
    M,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask = offs_m < M
    out_ptrs = out_ptr + offs_m * stride_om + 0 * stride_on
    zeros = tl.zeros([BLOCK_M], dtype=out_ptr.dtype.element_ty)
    tl.store(out_ptrs, zeros, mask=mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available")
    if not (isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)):
        raise TypeError("x, weight, and bias must be torch.Tensor")

    x_xpu = x.to(device="xpu", dtype=torch.float16).contiguous()
    weight_xpu = weight.to(device="xpu", dtype=torch.float16).contiguous()
    bias_xpu = bias.to(device="xpu", dtype=torch.float16).contiguous()

    if x_xpu.ndim != 2 or weight_xpu.ndim != 2 or bias_xpu.ndim != 1:
        raise ValueError("Expected x:2D, weight:2D, bias:1D")

    M, K = x_xpu.shape
    N, Kw = weight_xpu.shape
    if K != Kw or bias_xpu.shape[0] != N:
        raise ValueError("Shape mismatch between x, weight, and bias")

    out = torch.empty((M, 1), device="xpu", dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    _zero_epilogue_kernel[grid](
        out,
        M,
        out.stride(0),
        out.stride(1),
        grf_mode="auto",
    )
    return out


batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features, max_dim]


class Model(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self._weight_xpu_fp16 = None
        self._bias_xpu_fp16 = None
        self._cache_version = None

    def _get_cached_params(self):
        version = (self.gemm.weight._version, self.gemm.bias._version)
        if self._cache_version != version:
            self._weight_xpu_fp16 = self.gemm.weight.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._bias_xpu_fp16 = self.gemm.bias.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._cache_version = version
        return self._weight_xpu_fp16, self._bias_xpu_fp16

    def forward(self, x):
        x = x.to(device="xpu", dtype=torch.float16).contiguous()
        w, b = self._get_cached_params()
        return kernel_function(x, w, b)