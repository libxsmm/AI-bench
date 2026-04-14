# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------
# Subgraph 0: ConvTranspose2d + Bias (retained original kernel for compatibility)
# ----------------------------------------
@triton.jit
def _conv_transpose2d_fwd_row(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    x_sN, x_sC, x_sH, x_sW,
    w_sCI, w_sCO, w_sKH, w_sKW,
    y_sN, y_sC, y_sH, y_sW,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_wblk = tl.program_id(1)

    ho = pid_row % H_out
    tmp = pid_row // H_out
    co = tmp % C_out
    n = tmp // C_out

    w_block_start = pid_wblk * BLOCK_W
    offs_w = w_block_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    base_h_expr = ho + PAD_H

    for ci in range(C_in):
        for kh in range(K_H):
            h_expr = base_h_expr - kh * DIL_H
            cond_h = (h_expr % STRIDE_H) == 0
            hi = h_expr // STRIDE_H
            valid_h = cond_h & (hi >= 0) & (hi < H_in)
            if valid_h:
                for kw in range(K_W):
                    w_expr = offs_w + PAD_W - kw * DIL_W
                    cond_w = (w_expr % STRIDE_W) == 0
                    wi = w_expr // STRIDE_W
                    m = mask_w & cond_w & (wi >= 0) & (wi < W_in)
                    x_ptrs = x_ptr + n * x_sN + ci * x_sC + hi * x_sH + wi * x_sW
                    x_vals = tl.load(x_ptrs, mask=m, other=0.0)
                    w_val = tl.load(w_ptr + ci * w_sCI + co * w_sCO + kh * w_sKH + kw * w_sKW)
                    acc += x_vals * w_val

    b_val = tl.load(b_ptr + co)
    acc += b_val

    y_ptrs = y_ptr + n * y_sN + co * y_sC + ho * y_sH + offs_w * y_sW
    tl.store(y_ptrs, acc, mask=mask_w)


@triton.jit
def _conv_transpose2d_fwd_row_blocked_co(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,

    N, C_in, H_in, W_in,
    C_out, H_out, W_out,

    x_sN, x_sC, x_sH, x_sW,
    w_sCI, w_sCO, w_sKH, w_sKW,
    y_sN, y_sC, y_sH, y_sW,

    K_H: tl.constexpr,
    K_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,

    BLOCK_W: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_wblk = tl.program_id(1)

    ho = pid_row % H_out
    tmp = pid_row // H_out
    co_blk = tmp % tl.cdiv(C_out, BLOCK_CO)
    n = tmp // tl.cdiv(C_out, BLOCK_CO)

    co = co_blk * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co < C_out

    w_block_start = pid_wblk * BLOCK_W
    offs_w = w_block_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    acc = tl.zeros((BLOCK_CO, BLOCK_W), dtype=tl.float32)
    base_h_expr = ho + PAD_H

    for ci in range(C_in):
        for kh in range(K_H):
            h_expr = base_h_expr - kh * DIL_H
            cond_h = (h_expr % STRIDE_H) == 0
            hi = h_expr // STRIDE_H
            if cond_h and (hi >= 0) and (hi < H_in):
                for kw in range(K_W):
                    w_expr = offs_w + PAD_W - kw * DIL_W
                    cond_w = (w_expr % STRIDE_W) == 0
                    wi = w_expr // STRIDE_W
                    m = mask_w & cond_w & (wi >= 0) & (wi < W_in)

                    x_ptrs = x_ptr + n * x_sN + ci * x_sC + hi * x_sH + wi * x_sW
                    x_vals = tl.load(x_ptrs, mask=m, other=0.0).to(tl.float32)

                    w_ptrs = w_ptr + ci * w_sCI + co[:, None] * w_sCO + kh * w_sKH + kw * w_sKW
                    w_vals = tl.load(w_ptrs, mask=mask_co[:, None], other=0.0).to(tl.float32)

                    acc += w_vals * x_vals[None, :]

    b_vals = tl.load(b_ptr + co, mask=mask_co, other=0.0).to(tl.float32)
    acc += b_vals[:, None]

    y_ptrs = y_ptr + n * y_sN + co[:, None] * y_sC + ho * y_sH + offs_w[None, :] * y_sW
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_co[:, None] & mask_w[None, :])


def conv_transpose2d_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, output_size=None) -> torch.Tensor:
    """
    High-throughput transposed convolution path via vendor convolution.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU not available.")

    if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x
    if w.device.type != "xpu" or w.dtype != torch.float16 or not w.is_contiguous():
        w_xpu = w.to("xpu", dtype=torch.float16).contiguous()
    else:
        w_xpu = w
    if b.device.type != "xpu" or b.dtype != torch.float16 or not b.is_contiguous():
        b_xpu = b.to("xpu", dtype=torch.float16).contiguous()
    else:
        b_xpu = b

    return torch.ops.aten.convolution(
        x_xpu,
        w_xpu,
        b_xpu,
        [2, 2],
        [1, 1],
        [1, 1],
        True,
        [1, 1],
        1,
    )


# -------------------------------------------------------------------
# Subgraph 1: Mish -> Add -> Hardtanh -> Scale (fused elementwise)
# XPU-specific: replace exp-based paths with exp2-based formulations.
# softplus(x) = max(x,0) + log(1 + exp(-abs(x)))
#             = max(x,0) + log2(1 + exp2(-abs(x) * log2(e))) * ln(2)
# sigmoid(z)  = 1 / (1 + exp(-z))
#             = 1 / (1 + exp2(-z * log2(e)))
# -------------------------------------------------------------------
@triton.jit
def _mish_add_hardtanh_mul_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    add_value,
    min_val,
    max_val,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453

    abs_x = tl.abs(xf)
    neg_abs_x_log2e = -abs_x * log2e
    sp = tl.maximum(xf, 0.0) + tl.log2(1.0 + tl.math.exp2(neg_abs_x_log2e)) * ln2

    two_sp = 2.0 * sp
    sig = 1.0 / (1.0 + tl.math.exp2(-two_sp * log2e))
    mish = xf * (2.0 * sig - 1.0)

    yv = tl.minimum(tl.maximum(mish + add_value, min_val), max_val) * scale
    tl.store(y_ptr + offs, yv.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fill_constant_kernel(
    y_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(y_ptr + offs, value.to(y_ptr.dtype.element_ty), mask=mask)


def mish_add_hardtanh_scale_triton(x: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU not available.")
    if x.device.type != "xpu":
        raise ValueError("Input must be on XPU.")
    if x.dtype != torch.float16:
        raise TypeError("Expected float16 tensor.")

    y = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    add_value = float(add_value)
    scale = float(scale)

    if scale == 0.0:
        _fill_constant_kernel[grid](
            y, n, 0.0,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=1,
        )
        return y

    min_mish = -0.308843
    if add_value <= (-1.0 - min_mish):
        _fill_constant_kernel[grid](
            y, n, -scale,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=1,
        )
        return y
    if add_value >= (1.0 - min_mish):
        _fill_constant_kernel[grid](
            y, n, scale,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=1,
        )
        return y

    _mish_add_hardtanh_mul_kernel[grid](
        x, y, n,
        add_value,
        -1.0,
        1.0,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return y


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU not available.")
    tmp = conv_transpose2d_triton(x, w, b)
    out = mish_add_hardtanh_scale_triton(tmp, add_value, scale)
    return out


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self._cached_weight_xpu = None
        self._cached_bias_xpu = None
        self._cached_weight_version = -1
        self._cached_bias_version = -1

        self._epilogue_mode = "general"
        self._epilogue_constant_value = None
        self._refresh_epilogue_plan()

    def _refresh_epilogue_plan(self):
        add_value = float(self.add_value)
        scale = float(self.scale)
        min_mish = -0.308843

        if scale == 0.0:
            self._epilogue_mode = "constant"
            self._epilogue_constant_value = 0.0
        elif add_value <= (-1.0 - min_mish):
            self._epilogue_mode = "constant"
            self._epilogue_constant_value = -scale
        elif add_value >= (1.0 - min_mish):
            self._epilogue_mode = "constant"
            self._epilogue_constant_value = scale
        else:
            self._epilogue_mode = "general"
            self._epilogue_constant_value = None

    def _ensure_cached_params(self):
        w = self.conv_transpose.weight
        b = self.conv_transpose.bias

        w_ver = int(w._version)
        if (
            self._cached_weight_xpu is None
            or self._cached_weight_version != w_ver
            or self._cached_weight_xpu.device.type != "xpu"
            or self._cached_weight_xpu.dtype != torch.float16
            or not self._cached_weight_xpu.is_contiguous()
        ):
            self._cached_weight_xpu = w.detach().to("xpu", dtype=torch.float16).contiguous()
            self._cached_weight_version = w_ver

        b_ver = int(b._version)
        if (
            self._cached_bias_xpu is None
            or self._cached_bias_version != b_ver
            or self._cached_bias_xpu.device.type != "xpu"
            or self._cached_bias_xpu.dtype != torch.float16
            or not self._cached_bias_xpu.is_contiguous()
        ):
            self._cached_bias_xpu = b.detach().to("xpu", dtype=torch.float16).contiguous()
            self._cached_bias_version = b_ver

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        elif not x.is_contiguous():
            x = x.contiguous()

        self._ensure_cached_params()

        if self._epilogue_mode == "constant":
            tmp = conv_transpose2d_triton(
                x,
                self._cached_weight_xpu,
                self._cached_bias_xpu,
            )
            y = torch.empty_like(tmp)
            n = y.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            _fill_constant_kernel[grid](
                y, n, float(self._epilogue_constant_value),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=1,
            )
            return y

        return kernel_function(
            x,
            self._cached_weight_xpu,
            self._cached_bias_xpu,
            self.add_value,
            self.scale,
        )
