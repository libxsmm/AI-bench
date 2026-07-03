# ruff: noqa: E731
# KernelBench-compatible wrapper — Model class injected by codegen
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Original Triton kernel retained for compatibility/reference.
# -----------------------------------------------------------------------------
@triton.jit
def _conv3d_mul_reduce_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    mult_ptr,
    y_ptr,
    sum_ptr,
    sumsq_ptr,
    N,
    C_IN,
    C_OUT,
    D,
    H,
    W,
    D_OUT,
    H_OUT,
    W_OUT,
    BLOCK_HW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    n = pid2 // C_OUT
    co = pid2 % C_OUT

    s_start = pid0 * BLOCK_HW
    offs = s_start + tl.arange(0, BLOCK_HW)
    total_s = H_OUT * W_OUT
    mask_s = offs < total_s

    h_out = offs // W_OUT
    w_out = offs % W_OUT

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    b_val = tl.load(b_ptr + co).to(tl.float32)
    m_val = tl.load(mult_ptr + co).to(tl.float32)

    stride_ci = D * H * W
    stride_d = H * W
    stride_h = W

    for ci in range(C_IN):
        base_ci = ((n * C_IN) + ci) * stride_ci
        for kd in range(KD):
            d_in = pid1 + kd
            base_d = base_ci + d_in * stride_d
            for kh in range(KH):
                h_in = h_out + kh
                base_dh = base_d + h_in * stride_h
                for kw in range(KW):
                    w_in = w_out + kw
                    x_ptrs = x_ptr + base_dh + w_in
                    x_vals = tl.load(x_ptrs, mask=mask_s, other=0.0).to(tl.float32)
                    w_idx = (((co * C_IN + ci) * KD + kd) * KH + kh) * KW + kw
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                    acc += x_vals * w_val

    y_vals = (acc + b_val) * m_val

    y_stride_n = C_OUT * D_OUT * H_OUT * W_OUT
    y_stride_c = D_OUT * H_OUT * W_OUT
    y_stride_d = H_OUT * W_OUT
    y_base = n * y_stride_n + co * y_stride_c + pid1 * y_stride_d
    y_ptrs = y_ptr + y_base + offs
    tl.store(y_ptrs, y_vals.to(y_ptr.dtype.element_ty), mask=mask_s)

    y_valid = tl.where(mask_s, y_vals, 0.0)
    tile_sum = tl.sum(y_valid, axis=0)
    tile_sumsq = tl.sum(y_valid * y_valid, axis=0)
    stat_idx = n * C_OUT + co
    tl.atomic_add(sum_ptr + stat_idx, tile_sum)
    tl.atomic_add(sumsq_ptr + stat_idx, tile_sumsq)


# -----------------------------------------------------------------------------
# Original Triton kernel retained for compatibility/reference.
# -----------------------------------------------------------------------------
@triton.jit
def _instancenorm_clamp_mul_kernel(
    y_ptr,
    out_ptr,
    sum_ptr,
    sumsq_ptr,
    mult_ptr,
    N,
    C_OUT,
    D_OUT,
    H_OUT,
    W_OUT,
    eps,
    clamp_min,
    clamp_max,
    BLOCK_HW: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    n = pid2 // C_OUT
    co = pid2 % C_OUT

    s_start = pid0 * BLOCK_HW
    offs = s_start + tl.arange(0, BLOCK_HW)
    total_s = H_OUT * W_OUT
    mask_s = offs < total_s

    stat_idx = n * C_OUT + co
    sum_val = tl.load(sum_ptr + stat_idx)
    sumsq_val = tl.load(sumsq_ptr + stat_idx)
    count = D_OUT * H_OUT * W_OUT
    inv_count = 1.0 / count
    mean = sum_val * inv_count
    var = sumsq_val * inv_count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    y_stride_n = C_OUT * D_OUT * H_OUT * W_OUT
    y_stride_c = D_OUT * H_OUT * W_OUT
    y_stride_d = H_OUT * W_OUT
    y_base = n * y_stride_n + co * y_stride_c + pid1 * y_stride_d
    y_ptrs = y_ptr + y_base + offs
    y_vals = tl.load(y_ptrs, mask=mask_s, other=0.0).to(tl.float32)

    y_norm = (y_vals - mean) * inv_std
    y_clamped = tl.maximum(y_norm, clamp_min)
    y_clamped = tl.minimum(y_clamped, clamp_max)

    m_val2 = tl.load(mult_ptr + co).to(tl.float32)
    out_vals = y_clamped * m_val2

    out_ptrs = out_ptr + y_base + offs
    tl.store(out_ptrs, out_vals.to(out_ptr.dtype.element_ty), mask=mask_s)


# -----------------------------------------------------------------------------
# Original Triton kernel retained for compatibility/reference.
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=2),
    ],
    key=["W", "C"],
)
@triton.jit
def _reduce_max_c_dim1_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    C_CONST: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_nh = tl.program_id(2)

    n = pid_nh // H
    h = pid_nh % H
    d = pid_d

    start_w = pid_w * BLOCK_W
    offs_w = start_w + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    base = n * stride_n + d * stride_d + h * stride_h

    acc = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)

    for c in range(C_CONST):
        x_ptrs = x_ptr + base + c * stride_c + offs_w * stride_w
        vals = tl.load(x_ptrs, mask=mask_w, other=-float("inf")).to(tl.float32)
        acc = tl.maximum(acc, vals)

    y_base = n * out_stride_n + d * out_stride_d + h * out_stride_h
    y_ptrs = y_ptr + y_base + offs_w * out_stride_w
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_w)


def _post_kernel_autotune_configs():
    configs = [
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=2),
    ]
    return configs


# -----------------------------------------------------------------------------
# Fused post kernel:
#   input y0 = (conv + bias) * multiplier
#   stats are precomputed in fp32 outside Triton
#   compute instance norm + clamp + second mul + channel max
#   output: [N, D, H, W]
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_post_kernel_autotune_configs(),
    key=["W_OUT", "H_OUT", "D_OUT", "C_OUT"],
)
@triton.jit
def _instancenorm_clamp_mul_reduce_max_fp32stats_kernel(
    x_ptr,  # *[N, C, D, H, W] contiguous
    sum_ptr,  # *[N, C] fp32
    sumsq_ptr,  # *[N, C] fp32
    mult_ptr,  # *[C]
    out_ptr,  # *[N, D, H, W]
    C_OUT,
    D_OUT,
    H_OUT,
    W_OUT,
    eps,
    clamp_min,
    clamp_max,
    BLOCK_W: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_nh = tl.program_id(2)

    n = pid_nh // H_OUT
    h = pid_nh % H_OUT
    d = pid_d

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_OUT

    plane = H_OUT * W_OUT
    dhw = D_OUT * plane
    row_base = n * dhw + d * plane + h * W_OUT

    acc = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
    inv_count = 1.0 / (D_OUT * H_OUT * W_OUT)

    for c in range(C_CONST):
        stat_idx = n * C_OUT + c
        s = tl.load(sum_ptr + stat_idx)
        ss = tl.load(sumsq_ptr + stat_idx)

        mean = s * inv_count
        var = ss * inv_count - mean * mean
        var = tl.maximum(var, 0.0)
        inv_std = 1.0 / tl.sqrt(var + eps)

        m2 = tl.load(mult_ptr + c).to(tl.float32)

        x_base = (n * C_OUT + c) * dhw + d * plane + h * W_OUT
        vals = tl.load(x_ptr + x_base + offs_w, mask=mask_w, other=0.0).to(tl.float32)

        vals = (vals - mean) * inv_std
        vals = tl.maximum(vals, clamp_min)
        vals = tl.minimum(vals, clamp_max)
        vals = vals * m2

        acc = tl.maximum(acc, vals)

    tl.store(out_ptr + row_base + offs_w, acc.to(out_ptr.dtype.element_ty), mask=mask_w)


def kernel_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    eps: float = 1e-5,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups: int = 1,
):
    """
    Discovery optimization:
      - vendor-backed conv3d for the dominant compute stage
      - native XPU reductions for per-(n,c) sum/sumsq in fp32
      - single Triton fused post kernel for norm+clamp+mul+channel-max
    """
    assert (
        groups == 1
        and stride == (1, 1, 1)
        and padding == (0, 0, 0)
        and dilation == (1, 1, 1)
    )

    if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x

    if (
        weight.device.type != "xpu"
        or weight.dtype != torch.float16
        or not weight.is_contiguous()
    ):
        weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
    else:
        weight_xpu = weight

    if (
        bias.device.type != "xpu"
        or bias.dtype != torch.float16
        or not bias.is_contiguous()
    ):
        bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bias_xpu = bias

    mult_1d = multiplier.reshape(-1)
    if (
        mult_1d.device.type != "xpu"
        or mult_1d.dtype != torch.float16
        or not mult_1d.is_contiguous()
    ):
        mult_xpu = mult_1d.to("xpu", dtype=torch.float16).contiguous()
    else:
        mult_xpu = mult_1d

    N, C_in, D, H, W = x_xpu.shape
    C_out, C_in_w, KD, KH, KW = weight_xpu.shape
    assert C_in == C_in_w and (KD, KH, KW) == (3, 3, 3)
    assert bias_xpu.shape == (C_out,)
    assert mult_xpu.shape[0] == C_out

    conv = F.conv3d(
        x_xpu,
        weight_xpu,
        bias_xpu,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    y0 = (conv * mult_xpu.view(1, C_out, 1, 1, 1)).contiguous()

    _, _, D_out, H_out, W_out = y0.shape

    y0_fp32 = y0.to(torch.float32)
    sums = y0_fp32.sum(dim=(2, 3, 4)).contiguous()
    sumsqs = (y0_fp32 * y0_fp32).sum(dim=(2, 3, 4)).contiguous()

    out = torch.empty((N, D_out, H_out, W_out), device=y0.device, dtype=y0.dtype)

    grid_post = lambda META: (triton.cdiv(W_out, META["BLOCK_W"]), D_out, N * H_out)
    _instancenorm_clamp_mul_reduce_max_fp32stats_kernel[grid_post](
        y0,
        sums,
        sumsqs,
        mult_xpu,
        out,
        C_out,
        D_out,
        H_out,
        W_out,
        eps,
        clamp_min,
        clamp_max,
        C_CONST=C_out,
    )

    return out


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        multiplier_shape,
        clamp_min,
        clamp_max,
    ]


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        multiplier_shape,
        clamp_min,
        clamp_max,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.ones(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        elif not x.is_contiguous():
            x = x.contiguous()

        if (
            self.conv.weight.device.type != "xpu"
            or self.conv.weight.dtype != torch.float16
        ):
            self.conv.weight.data = self.conv.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        elif not self.conv.weight.is_contiguous():
            self.conv.weight.data = self.conv.weight.data.contiguous()

        if self.conv.bias is not None:
            if (
                self.conv.bias.device.type != "xpu"
                or self.conv.bias.dtype != torch.float16
            ):
                self.conv.bias.data = self.conv.bias.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()
            elif not self.conv.bias.is_contiguous():
                self.conv.bias.data = self.conv.bias.data.contiguous()

        if (
            self.multiplier.device.type != "xpu"
            or self.multiplier.dtype != torch.float16
        ):
            self.multiplier.data = self.multiplier.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        elif not self.multiplier.is_contiguous():
            self.multiplier.data = self.multiplier.data.contiguous()

        return kernel_function(
            x,
            self.conv.weight,
            self.conv.bias,
            self.multiplier,
            self.clamp_min,
            self.clamp_max,
        )
