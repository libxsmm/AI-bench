# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------
# Original Triton GEMM kernel kept in the codebase for reference.
# Discovery-stage execution path prefers vendor GEMM.
# ---------------------------------------------------------------------
_linear_configs = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=16),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=16),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=_linear_configs, key=['M', 'N', 'K'])
@triton.jit
def _linear_fwd_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    ADD_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    if ADD_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias_vals[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def _linear_forward(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
    if not (isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(bias, torch.Tensor)):
        raise TypeError("x, w, bias must be Tensors")
    if x.device != w.device or x.device != bias.device:
        raise ValueError("x, w, bias must be on same device")
    if x.device.type != 'xpu':
        raise RuntimeError(f"Linear kernel requires 'xpu' device, got {x.device}")
    if x.ndim != 2 or w.ndim != 2 or bias.ndim != 1:
        raise ValueError("Shapes: x[M, K], w[N, K], bias[N]")

    M, K = x.shape
    Nw, Kw = w.shape
    if K != Kw:
        raise ValueError(f"Incompatible K: x.K={K}, w.K={Kw}")
    if bias.shape[0] != Nw:
        raise ValueError(f"Bias shape {bias.shape} does not match w rows {Nw}")
    N = Nw

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = w.stride()
    stride_ym, stride_yn = y.stride()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _linear_fwd_kernel[grid](
        x, w, bias, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        True
    )
    return y


# ---------------------------------------------------------------------
# Pointwise epilogue kernel
# ---------------------------------------------------------------------
@triton.jit
def _sigmoid_stable(x):
    e = tl.exp(-tl.abs(x))
    return tl.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


_swish_configs = [
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
]


@triton.autotune(configs=_swish_configs, key=['N'])
@triton.jit
def _fused_swish_div_clamp_tanh_clamp_kernel(
    x_ptr, y_ptr, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    s = _sigmoid_stable(x)
    y = 0.5 * x * s

    y = tl.maximum(tl.minimum(y, 1.0), -1.0)

    y2 = 2.0 * y
    y = 2.0 * _sigmoid_stable(y2) - 1.0

    # Final clamp is mathematically redundant because tanh(z) in (-1, 1).
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def _swish_forward(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.device.type != "xpu":
        raise RuntimeError(f"Swish kernel requires 'xpu', got {x.device}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Unsupported dtype")
    if not x.is_contiguous():
        x = x.contiguous()

    y = torch.empty_like(x)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta['BLOCK_SIZE']),)

    _fused_swish_div_clamp_tanh_clamp_kernel[grid](x, y, n)
    return y


def _ensure_xpu_fp16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "xpu" or t.dtype != torch.float16:
        t = t.to("xpu", dtype=torch.float16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def kernel_function(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
    if not (isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(bias, torch.Tensor)):
        raise TypeError("Expected Tensors x, w, bias")

    x_xpu = _ensure_xpu_fp16_contiguous(x)
    w_xpu = _ensure_xpu_fp16_contiguous(w)
    bias_xpu = _ensure_xpu_fp16_contiguous(bias)

    y0 = torch.nn.functional.linear(x_xpu, w_xpu, bias_xpu)
    if not y0.is_contiguous():
        y0 = y0.contiguous()
    y1 = _swish_forward(y0)
    return y1


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.bias = bias

        self._packed_weight = None
        self._packed_bias = None
        self._packed_weight_version = -1
        self._packed_bias_version = -1

    def _ensure_packed_params(self):
        w = self.gemm.weight
        w_ver = int(w._version)
        if (
            self._packed_weight is None
            or self._packed_weight_version != w_ver
            or self._packed_weight.device.type != "xpu"
            or self._packed_weight.dtype != torch.float16
            or not self._packed_weight.is_contiguous()
            or tuple(self._packed_weight.shape) != tuple(w.shape)
        ):
            self._packed_weight = _ensure_xpu_fp16_contiguous(w.detach())
            self._packed_weight_version = w_ver

        b = self.gemm.bias
        if b is not None:
            b_ver = int(b._version)
            if (
                self._packed_bias is None
                or self._packed_bias_version != b_ver
                or self._packed_bias.device.type != "xpu"
                or self._packed_bias.dtype != torch.float16
                or not self._packed_bias.is_contiguous()
                or tuple(self._packed_bias.shape) != tuple(b.shape)
            ):
                self._packed_bias = _ensure_xpu_fp16_contiguous(b.detach())
                self._packed_bias_version = b_ver
        else:
            self._packed_bias = None
            self._packed_bias_version = -1

    def forward(self, x):
        x_xpu = _ensure_xpu_fp16_contiguous(x)
        self._ensure_packed_params()
        return kernel_function(x_xpu, self._packed_weight, self._packed_bias)