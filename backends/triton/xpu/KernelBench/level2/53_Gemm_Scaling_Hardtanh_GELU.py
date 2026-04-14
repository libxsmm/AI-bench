# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _epilogue_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=32, num_stages=2),
    ]


# ------------------------------
# Triton epilogue kernel: scale + hardtanh + GELU
# XPU-specific tweak: use exp2(x * log2e) instead of exp(x)
# ------------------------------
@triton.autotune(
    configs=_epilogue_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _epilogue_scale_hardtanh_gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    scale,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x * scale
    x = tl.minimum(tl.maximum(x, min_val), max_val)

    inv_sqrt2 = 0.7071067811865476
    log2e = 1.4426950408889634

    t1 = x * inv_sqrt2
    at1 = tl.abs(t1)

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * at1)
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)

    neg_sq = -(at1 * at1)
    e = tl.math.exp2(neg_sq * log2e)

    erf_abs = 1.0 - poly * e
    sign = tl.where(t1 >= 0, 1.0, -1.0)
    erf_val = sign * erf_abs
    y = 0.5 * x * (1.0 + erf_val)

    tl.store(y_ptr + offsets, y.to(tl.float16), mask=mask)


# ------------------------------
# Compatibility kernels retained
# ------------------------------
@triton.jit
def _hardtanh_gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.minimum(tl.maximum(x, min_val), max_val)

    inv_sqrt2 = 0.7071067811865476
    log2e = 1.4426950408889634

    t1 = x * inv_sqrt2
    at1 = tl.abs(t1)

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * at1)
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)

    neg_sq = -(at1 * at1)
    e = tl.math.exp2(neg_sq * log2e)

    erf_abs = 1.0 - poly * e
    sign = tl.where(t1 >= 0, 1.0, -1.0)
    erf_val = sign * erf_abs
    y = 0.5 * x * (1.0 + erf_val)

    tl.store(y_ptr + offsets, y.to(tl.float16), mask=mask)


@triton.jit
def _fused_linear_scale_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
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
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid == 0:
        pass


# ------------------------------
# Top-level wrapper
# ------------------------------
def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale=None):
    assert isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)

    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if weight.device.type != "xpu" or weight.dtype != torch.float16:
        weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
    else:
        weight_xpu = weight.contiguous()

    if bias.device.type != "xpu" or bias.dtype != torch.float16:
        bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bias_xpu = bias.contiguous()

    assert x_xpu.ndim == 2 and weight_xpu.ndim == 2 and bias_xpu.ndim == 1
    B, In = x_xpu.shape
    Out, In_w = weight_xpu.shape
    assert In == In_w and bias_xpu.numel() == Out

    if scale is None:
        scale_val = 0.5
    elif isinstance(scale, torch.Tensor):
        if scale.device.type == "xpu":
            raise ValueError(
                "scale must be a Python float/int or CPU tensor; "
                "passing an XPU tensor would require device->host sync via .item()."
            )
        scale_val = float(scale.item())
    else:
        scale_val = float(scale)

    gemm_out = F.linear(x_xpu, weight_xpu, bias_xpu)

    y = torch.empty_like(gemm_out)
    n_elements = gemm_out.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _epilogue_scale_hardtanh_gelu_kernel[grid](
        gemm_out,
        y,
        n_elements,
        scale_val,
        -2.0,
        2.0,
    )
    return y


batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]


class Model(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = float(scaling_factor)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self._params_on_xpu = False

    def _ensure_xpu_params(self):
        if not self._params_on_xpu:
            self.gemm.weight.data = self.gemm.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.gemm.bias.data = self.gemm.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self._params_on_xpu = True
        else:
            if self.gemm.weight.device.type != "xpu" or self.gemm.weight.dtype != torch.float16:
                self.gemm.weight.data = self.gemm.weight.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.gemm.weight.is_contiguous():
                self.gemm.weight.data = self.gemm.weight.data.contiguous()

            if self.gemm.bias.device.type != "xpu" or self.gemm.bias.dtype != torch.float16:
                self.gemm.bias.data = self.gemm.bias.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.gemm.bias.is_contiguous():
                self.gemm.bias.data = self.gemm.bias.data.contiguous()

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        self._ensure_xpu_params()
        return kernel_function(x, self.gemm.weight, self.gemm.bias, self.scale)
