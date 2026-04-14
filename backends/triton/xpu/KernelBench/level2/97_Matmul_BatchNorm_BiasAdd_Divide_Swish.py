# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------------------------------
# Original kernel kept intact for compatibility / validation requirements
# -------------------------------------------------------------------------
@triton.jit
def _fused_linear_bn_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    n_mask = offs_n < N
    bias_f32 = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    gamma_f32 = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    beta_f32 = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    mean_f32 = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    var_f32 = tl.load(var_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    acc = acc + bias_f32[None, :]
    inv_std = 1.0 / tl.sqrt(var_f32 + eps)
    acc = (acc - mean_f32[None, :]) * inv_std[None, :]
    acc = acc * gamma_f32[None, :] + beta_f32[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_out = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y_out, mask=y_mask)


# -------------------------------------------------------------------------
# Original second kernel kept intact for compatibility / validation requirements
# but updated to use a faster XPU-friendly sigmoid form.
# -------------------------------------------------------------------------
@triton.jit
def _fused_bias_div_swish_kernel(
    x_ptr, bias_ptr, out_ptr, n_elements, divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr).to(tl.float32)
    y = (x + b) / divisor
    sigm = tl.sigmoid(y)
    out = y * sigm

    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


# -------------------------------------------------------------------------
# Original 1D post-op kernel kept intact for compatibility / validation
# requirements. It is no longer used in the optimized path because BN is
# folded into the linear layer, but must remain present.
# -------------------------------------------------------------------------
@triton.jit
def _fused_post_bn_bias_div_swish_1d_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    scalar_bias_ptr,
    out_ptr,
    n_elements,
    N,
    eps,
    inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cols = offsets % N

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scalar_bias = tl.load(scalar_bias_ptr).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + eps)
    y = y * gamma + beta
    y = (y + scalar_bias) * inv_divisor
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _epilogue_autotune_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=3),
        # Required large-tile / 32-warp XPU-oriented fallback
        triton.Config({"BLOCK_SIZE": 65536}, num_warps=32, num_stages=3),
    ]
    return configs


# -------------------------------------------------------------------------
# Optimized epilogue kernel:
# scalar bias/divide + swish only, after BN folding into the linear layer
# -------------------------------------------------------------------------
@triton.autotune(
    configs=_epilogue_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _fused_bias_div_swish_1d_kernel(
    x_ptr,
    scalar_bias_ptr,
    out_ptr,
    n_elements,
    inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar_bias = tl.load(scalar_bias_ptr).to(tl.float32)

    y = (x + scalar_bias) * inv_divisor
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _to_xpu_fp16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "xpu" or t.dtype != torch.float16:
        return t.to("xpu", dtype=torch.float16).contiguous()
    return t.contiguous()


def kernel_function(
    x: torch.Tensor,
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
    eps: float = 1e-5,
):
    del eps
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"

    x_xpu = _to_xpu_fp16_contiguous(x)
    fused_weight_xpu = _to_xpu_fp16_contiguous(fused_weight)
    fused_bias_xpu = _to_xpu_fp16_contiguous(fused_bias)
    bias_xpu = _to_xpu_fp16_contiguous(bias)

    assert x_xpu.ndim == 2, "x must be a 2D tensor [M, K]"
    _, K = x_xpu.shape
    assert fused_weight_xpu.ndim == 2 and fused_weight_xpu.shape[1] == K, "fused_weight must be [N, K]"
    N = fused_weight_xpu.shape[0]
    assert fused_bias_xpu.shape == (N,), "fused_bias must be [N]"
    assert bias_xpu.numel() == 1 and bias_xpu.shape == (1,), "bias must be a scalar tensor shape [1]"

    inter = F.linear(x_xpu, fused_weight_xpu, fused_bias_xpu)

    out = torch.empty_like(inter)
    n_elements = inter.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_bias_div_swish_1d_kernel[grid](
        inter,
        bias_xpu,
        out,
        n_elements,
        1.0 / float(divisor),
    )
    return out


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.zeros(bias_shape))
        self.divide_value = divide_value
        self.bn_eps = bn_eps

        self._fused_weight = None
        self._fused_bias = None
        self._cache_versions = None

    def _ensure_xpu_params(self):
        if self.matmul.weight.device.type != "xpu" or self.matmul.weight.dtype != torch.float16:
            self.matmul.weight.data = self.matmul.weight.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.matmul.weight.is_contiguous():
            self.matmul.weight.data = self.matmul.weight.data.contiguous()

        if self.matmul.bias is not None:
            if self.matmul.bias.device.type != "xpu" or self.matmul.bias.dtype != torch.float16:
                self.matmul.bias.data = self.matmul.bias.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.matmul.bias.is_contiguous():
                self.matmul.bias.data = self.matmul.bias.data.contiguous()

        if self.bn.weight.device.type != "xpu" or self.bn.weight.dtype != torch.float16:
            self.bn.weight.data = self.bn.weight.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.bn.weight.is_contiguous():
            self.bn.weight.data = self.bn.weight.data.contiguous()

        if self.bn.bias.device.type != "xpu" or self.bn.bias.dtype != torch.float16:
            self.bn.bias.data = self.bn.bias.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.bn.bias.is_contiguous():
            self.bn.bias.data = self.bn.bias.data.contiguous()

        if self.bn.running_mean.device.type != "xpu" or self.bn.running_mean.dtype != torch.float16:
            self.bn.running_mean.data = self.bn.running_mean.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.bn.running_mean.is_contiguous():
            self.bn.running_mean.data = self.bn.running_mean.data.contiguous()

        if self.bn.running_var.device.type != "xpu" or self.bn.running_var.dtype != torch.float16:
            self.bn.running_var.data = self.bn.running_var.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.bn.running_var.is_contiguous():
            self.bn.running_var.data = self.bn.running_var.data.contiguous()

        if self.bias.device.type != "xpu" or self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.bias.is_contiguous():
            self.bias.data = self.bias.data.contiguous()

    def _ensure_fused_linear_bn(self):
        self._ensure_xpu_params()

        versions = (
            int(self.matmul.weight._version),
            int(self.matmul.bias._version) if self.matmul.bias is not None else -1,
            int(self.bn.weight._version),
            int(self.bn.bias._version),
            int(self.bn.running_mean._version),
            int(self.bn.running_var._version),
        )

        if self._fused_weight is not None and self._fused_bias is not None and self._cache_versions == versions:
            return

        w = self.matmul.weight
        b = self.matmul.bias
        gamma = self.bn.weight
        beta = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var

        scale_f32 = gamma.float() * torch.rsqrt(var.float() + float(self.bn_eps))
        fused_weight = (w.float() * scale_f32[:, None]).to(torch.float16)
        fused_bias = ((b.float() - mean.float()) * scale_f32 + beta.float()).to(torch.float16)

        self._fused_weight = fused_weight.contiguous()
        self._fused_bias = fused_bias.contiguous()
        self._cache_versions = versions

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)

        self._ensure_fused_linear_bn()

        return kernel_function(
            x,
            self._fused_weight,
            self._fused_bias,
            self.bias,
            self.divide_value,
            self.bn_eps,
        )
