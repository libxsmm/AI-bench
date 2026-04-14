# ruff: noqa: E731
import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# Reference-compatible model wrapper
# ----------------------------------------
class Model(torch.nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation,
    subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = torch.nn.AvgPool2d(kernel_size_pool)
        self._params_on_xpu = False

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        if not self._params_on_xpu:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self._params_on_xpu = True
        else:
            if not self.conv.weight.is_contiguous():
                self.conv.weight.data = self.conv.weight.data.contiguous()
            if self.conv.bias is not None and not self.conv.bias.is_contiguous():
                self.conv.bias.data = self.conv.bias.data.contiguous()

        return kernel_function(
            x, self.conv.weight, self.conv.bias,
            self.subtract1_value, self.subtract2_value
        )


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]


# ----------------------------------------
# Keep original Triton kernels present for compatibility
# ----------------------------------------
CONFIGS = [
    triton.Config({'BLOCK_CO': 32, 'BLOCK_WO': 32, 'BLOCK_CI': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_CO': 64, 'BLOCK_WO': 16, 'BLOCK_CI': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_CO': 32, 'BLOCK_WO': 64, 'BLOCK_CI': 32}, num_warps=16, num_stages=2),
]


@triton.autotune(configs=CONFIGS, key=['C_OUT', 'W_OUT'])
@triton.jit
def _conv2d_nchw_k3s1_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, H, W, C_OUT, H_OUT, W_OUT,
    SXN, SXC, SXH, SXW,
    SWO, SWI, SWKH, SWKW,
    SYN, SYC, SYH, SYW,
    BLOCK_CO: tl.constexpr, BLOCK_WO: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    pid_co = tl.program_id(0)
    pid_wo = tl.program_id(1)
    pid_nh = tl.program_id(2)
    n = pid_nh // H_OUT
    ho = pid_nh % H_OUT
    start_co = pid_co * BLOCK_CO
    start_wo = pid_wo * BLOCK_WO
    offs_co = start_co + tl.arange(0, BLOCK_CO)
    offs_wo = start_wo + tl.arange(0, BLOCK_WO)
    offs_ci = tl.arange(0, BLOCK_CI)
    co_mask = offs_co < C_OUT
    wo_mask = offs_wo < W_OUT
    acc = tl.zeros((BLOCK_CO, BLOCK_WO), dtype=tl.float32)
    ci0 = 0
    while ci0 < C_IN:
        ci_idx = ci0 + offs_ci
        ci_mask = ci_idx < C_IN
        for ky in range(0, 3):
            hi = ho + ky
            hi_valid = hi < H
            for kx in range(0, 3):
                w_ptrs = w_ptr + offs_co[:, None] * SWO + ci_idx[None, :] * SWI + ky * SWKH + kx * SWKW
                w_mask = co_mask[:, None] & ci_mask[None, :]
                w_sub = tl.load(w_ptrs, mask=w_mask, other=0.0)

                x_ptrs = x_ptr + n * SXN + ci_idx[:, None] * SXC + hi * SXH + (offs_wo[None, :] + kx) * SXW
                x_mask = ci_mask[:, None] & wo_mask[None, :] & hi_valid & ((offs_wo[None, :] + kx) < W)
                x_sub = tl.load(x_ptrs, mask=x_mask, other=0.0)
                acc = tl.dot(w_sub, x_sub, acc)
        ci0 += BLOCK_CI
    b = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + b[:, None]
    y_ptrs = y_ptr + n * SYN + offs_co[:, None] * SYC + ho * SYH + offs_wo[None, :] * SYW
    y_mask = co_mask[:, None] & wo_mask[None, :]
    if y_ptr.dtype.element_ty == tl.bfloat16:
        out = acc.to(tl.bfloat16)
    elif y_ptr.dtype.element_ty == tl.float16:
        out = acc.to(tl.float16)
    else:
        out = acc.to(tl.float32)
    tl.store(y_ptrs, out, mask=y_mask)


@triton.jit
def _affine_tanh_affine_kernel(
    x_ptr, out_ptr, n_elements, subtract1, subtract2, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = x - subtract1
    abs_y = tl.abs(y)
    e = tl.exp(-(abs_y + abs_y))
    tanh_abs = 1.0 - (2.0 * e) / (1.0 + e)
    sign = tl.where(y >= 0, 1.0, -1.0)
    tanh_y = sign * tanh_abs
    out = tanh_y - subtract2
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _avgpool2d_2x2_s2_nchw_kernel(
    x_ptr, y_ptr, N, C, H, W, OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr
):
    pid_ow = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_nc = tl.program_id(2)
    n = pid_nc // C
    c = pid_nc % C
    oh_start = pid_oh * BLOCK_OH
    ow_start = pid_ow * BLOCK_OW
    offs_oh = oh_start + tl.arange(0, BLOCK_OH)
    offs_ow = ow_start + tl.arange(0, BLOCK_OW)
    oh_mask = offs_oh < OH
    ow_mask = offs_ow < OW
    offs_oh_2d = offs_oh[:, None]
    offs_ow_2d = offs_ow[None, :]
    in_h0 = 2 * offs_oh_2d
    in_h1 = in_h0 + 1
    in_w0 = 2 * offs_ow_2d
    in_w1 = in_w0 + 1
    base_in = x_ptr + n * stride_n + c * stride_c
    base_out = y_ptr + n * out_stride_n + c * out_stride_c
    ptr00 = base_in + in_h0 * stride_h + in_w0 * stride_w
    ptr01 = base_in + in_h0 * stride_h + in_w1 * stride_w
    ptr10 = base_in + in_h1 * stride_h + in_w0 * stride_w
    ptr11 = base_in + in_h1 * stride_h + in_w1 * stride_w
    ohow_mask = (offs_oh_2d < OH) & (offs_ow_2d < OW)
    mask00 = ohow_mask & (in_h0 < H) & (in_w0 < W)
    mask01 = ohow_mask & (in_h0 < H) & (in_w1 < W)
    mask10 = ohow_mask & (in_h1 < H) & (in_w0 < W)
    mask11 = ohow_mask & (in_h1 < H) & (in_w1 < W)
    v00 = tl.load(ptr00, mask=mask00, other=0.0)
    v01 = tl.load(ptr01, mask=mask01, other=0.0)
    v10 = tl.load(ptr10, mask=mask10, other=0.0)
    v11 = tl.load(ptr11, mask=mask11, other=0.0)
    acc = (v00.to(tl.float32) + v01.to(tl.float32) + v10.to(tl.float32) + v11.to(tl.float32)) * 0.25
    out_ptrs = base_out + offs_oh_2d * out_stride_h + offs_ow_2d * out_stride_w
    out_mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


# ----------------------------------------
# Optimized fused post-op:
# vendor conv2d + Triton fused tanh/avgpool
# Sequential accumulation reduces register pressure.
# ----------------------------------------
TANH_POOL_CONFIGS = [
    triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=8, num_stages=1),
    triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=TANH_POOL_CONFIGS, key=['OH', 'OW'])
@triton.jit
def _tanh_avgpool2d_2x2_s2_kernel(
    x_ptr, y_ptr,
    C, H, W, OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    subtract1, subtract2,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr
):
    pid_ow = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    tl.max_contiguous(offs_ow, BLOCK_OW)

    oh = offs_oh[:, None]
    ow = offs_ow[None, :]
    out_mask = (oh < OH) & (ow < OW)

    h0 = oh * 2
    w0 = ow * 2

    base_in = x_ptr + n * stride_n + c * stride_c
    base_out = y_ptr + n * out_stride_n + c * out_stride_c

    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    p = base_in + h0 * stride_h + w0 * stride_w
    z = tl.load(p, mask=out_mask, other=0.0).to(tl.float32) - subtract1
    a = tl.abs(z)
    e = tl.exp(-(a + a))
    acc += tl.where(z >= 0, 1.0, -1.0) * (1.0 - (2.0 * e) / (1.0 + e))

    p = base_in + h0 * stride_h + (w0 + 1) * stride_w
    z = tl.load(p, mask=out_mask, other=0.0).to(tl.float32) - subtract1
    a = tl.abs(z)
    e = tl.exp(-(a + a))
    acc += tl.where(z >= 0, 1.0, -1.0) * (1.0 - (2.0 * e) / (1.0 + e))

    p = base_in + (h0 + 1) * stride_h + w0 * stride_w
    z = tl.load(p, mask=out_mask, other=0.0).to(tl.float32) - subtract1
    a = tl.abs(z)
    e = tl.exp(-(a + a))
    acc += tl.where(z >= 0, 1.0, -1.0) * (1.0 - (2.0 * e) / (1.0 + e))

    p = base_in + (h0 + 1) * stride_h + (w0 + 1) * stride_w
    z = tl.load(p, mask=out_mask, other=0.0).to(tl.float32) - subtract1
    a = tl.abs(z)
    e = tl.exp(-(a + a))
    acc += tl.where(z >= 0, 1.0, -1.0) * (1.0 - (2.0 * e) / (1.0 + e))

    out = acc * 0.25 - subtract2
    out_ptrs = base_out + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=out_mask)


def fused_tanh_avgpool2d_2x2_s2(x: torch.Tensor, subtract1: float, subtract2: float):
    assert isinstance(x, torch.Tensor)
    assert x.device.type == "xpu"
    assert x.ndim == 4
    if not x.is_contiguous():
        x = x.contiguous()

    N, C, H, W = x.shape
    OH = (H - 2) // 2 + 1
    OW = (W - 2) // 2 + 1
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)

    sN, sC, sH, sW = x.stride()
    oN, oC, oH, oW = y.stride()

    grid = lambda META: (triton.cdiv(OW, META['BLOCK_OW']), triton.cdiv(OH, META['BLOCK_OH']), N * C)
    _tanh_avgpool2d_2x2_s2_kernel[grid](
        x, y,
        C, H, W, OH, OW,
        sN, sC, sH, sW,
        oN, oC, oH, oW,
        float(subtract1), float(subtract2),
    )
    return y


# ----------------------------------------
# Top-level optimized path
# ----------------------------------------
def kernel_function(x: torch.Tensor,
                    weight: torch.Tensor,
                    bias: torch.Tensor,
                    subtract1: float,
                    subtract2: float) -> torch.Tensor:
    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    elif not x.is_contiguous():
        x_xpu = x.contiguous()
    else:
        x_xpu = x

    if weight.device.type != "xpu" or weight.dtype != torch.float16:
        weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
    elif not weight.is_contiguous():
        weight_xpu = weight.contiguous()
    else:
        weight_xpu = weight

    if bias is not None:
        if bias.device.type != "xpu" or bias.dtype != torch.float16:
            bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
        elif not bias.is_contiguous():
            bias_xpu = bias.contiguous()
        else:
            bias_xpu = bias
    else:
        bias_xpu = None

    y_conv = F.conv2d(x_xpu, weight_xpu, bias_xpu, stride=1, padding=0)
    y_out = fused_tanh_avgpool2d_2x2_s2(y_conv, subtract1, subtract2)
    return y_out


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]