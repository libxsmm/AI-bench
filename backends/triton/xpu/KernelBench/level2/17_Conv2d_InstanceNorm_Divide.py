# ruff: noqa: E731
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------
# Original kernel kept intact
# -------------------------------
@triton.jit
def _conv2d_nchw_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, H, W, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_nh = tl.program_id(1)
    pid_co = tl.program_id(2)

    num_h_tiles = tl.cdiv(H_out, BLOCK_H)
    n_idx = pid_nh // num_h_tiles
    h_tile_idx = pid_nh - n_idx * num_h_tiles

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ho = h_tile_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_wo = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    co_mask = offs_co < Cout
    ho_mask = offs_ho < H_out
    wo_mask = offs_wo < W_out
    hw_mask = ho_mask[:, None] & wo_mask[None, :]

    acc = tl.zeros((BLOCK_CO, BLOCK_H, BLOCK_W), dtype=tl.float32)

    x_base_n = x_ptr + n_idx * x_stride_n

    for ci in tl.range(0, Cin):
        x_base_nc = x_base_n + ci * x_stride_c
        w_base_c = w_ptr + ci * w_stride_ci

        for kh in tl.static_range(0, K):
            in_h = offs_ho * STRIDE_H - PAD_H + kh * DIL_H
            in_h_ok = (in_h >= 0) & (in_h < H)

            for kw in tl.static_range(0, K):
                in_w = offs_wo * STRIDE_W - PAD_W + kw * DIL_W
                in_w_ok = (in_w >= 0) & (in_w < W)

                load_mask = hw_mask & in_h_ok[:, None] & in_w_ok[None, :]
                x_ptrs = x_base_nc + in_h[:, None] * x_stride_h + in_w[None, :] * x_stride_w
                x_tile = tl.load(x_ptrs, mask=load_mask, other=0.0).to(tl.float32)

                w_ptrs = w_base_c + offs_co * w_stride_co + kh * w_stride_kh + kw * w_stride_kw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                acc += w_vec[:, None, None] * x_tile[None, :, :]

    bias = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias[:, None, None]

    y_ptrs = (
        y_ptr
        + n_idx * y_stride_n
        + offs_co[:, None, None] * y_stride_c
        + offs_ho[None, :, None] * y_stride_h
        + offs_wo[None, None, :] * y_stride_w
    )
    store_mask = co_mask[:, None, None] & hw_mask[None, :, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=store_mask)


# ----------------------------------------
# Original kernel kept intact
# ----------------------------------------
@triton.jit
def _instance_norm2d_scale_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    eps, scale,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * stride_n + c * stride_c
    HW = H * W

    sum_val = tl.zeros((1,), dtype=tl.float32)
    sum_sq = tl.zeros((1,), dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h_idx = offs // W
        w_idx = offs - h_idx * W
        ptrs = x_ptr + base + h_idx * stride_h + w_idx * stride_w
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    hw_f = tl.full((1,), HW, dtype=tl.float32)
    mean = sum_val / hw_f
    var = sum_sq / hw_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    fused_scale = tl.full((1,), scale, dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h_idx = offs // W
        w_idx = offs - h_idx * W
        ptrs = x_ptr + base + h_idx * stride_h + w_idx * stride_w
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std * fused_scale
        out_ptrs = y_ptr + base + h_idx * stride_h + w_idx * stride_w
        tl.store(out_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------
# New fusion-oriented helper kernels
# ---------------------------------------------------------
@triton.jit
def _conv2d_nchw_bias_stats_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, sum_ptr, sumsq_ptr,
    N, Cin, Cout, H, W, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    stats_stride_n, stats_stride_c, stats_stride_t,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_nh = tl.program_id(1)
    pid_co = tl.program_id(2)

    num_h_tiles = tl.cdiv(H_out, BLOCK_H)
    num_w_tiles = tl.cdiv(W_out, BLOCK_W)

    n_idx = pid_nh // num_h_tiles
    h_tile_idx = pid_nh - n_idx * num_h_tiles

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ho = h_tile_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_wo = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    co_mask = offs_co < Cout
    ho_mask = offs_ho < H_out
    wo_mask = offs_wo < W_out
    hw_mask = ho_mask[:, None] & wo_mask[None, :]

    acc = tl.zeros((BLOCK_CO, BLOCK_H, BLOCK_W), dtype=tl.float32)
    x_base_n = x_ptr + n_idx * x_stride_n

    for ci in tl.range(0, Cin):
        x_base_nc = x_base_n + ci * x_stride_c
        w_base_c = w_ptr + ci * w_stride_ci

        for kh in tl.static_range(0, K):
            in_h = offs_ho * STRIDE_H - PAD_H + kh * DIL_H
            in_h_ok = (in_h >= 0) & (in_h < H)

            for kw in tl.static_range(0, K):
                in_w = offs_wo * STRIDE_W - PAD_W + kw * DIL_W
                in_w_ok = (in_w >= 0) & (in_w < W)

                load_mask = hw_mask & in_h_ok[:, None] & in_w_ok[None, :]
                x_ptrs = x_base_nc + in_h[:, None] * x_stride_h + in_w[None, :] * x_stride_w
                x_tile = tl.load(x_ptrs, mask=load_mask, other=0.0).to(tl.float32)

                w_ptrs = w_base_c + offs_co * w_stride_co + kh * w_stride_kh + kw * w_stride_kw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                acc += w_vec[:, None, None] * x_tile[None, :, :]

    bias = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias[:, None, None]

    y_ptrs = (
        y_ptr
        + n_idx * y_stride_n
        + offs_co[:, None, None] * y_stride_c
        + offs_ho[None, :, None] * y_stride_h
        + offs_wo[None, None, :] * y_stride_w
    )
    store_mask = co_mask[:, None, None] & hw_mask[None, :, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=store_mask)

    valid = store_mask.to(tl.float32)
    acc_masked = acc * valid
    part_sum = tl.sum(tl.sum(acc_masked, axis=2), axis=1)
    part_sumsq = tl.sum(tl.sum(acc_masked * acc_masked, axis=2), axis=1)

    tile_linear = h_tile_idx * num_w_tiles + pid_w
    stats_ptrs = (
        n_idx * stats_stride_n
        + offs_co * stats_stride_c
        + tile_linear * stats_stride_t
    )

    tl.store(sum_ptr + stats_ptrs, part_sum, mask=co_mask)
    tl.store(sumsq_ptr + stats_ptrs, part_sumsq, mask=co_mask)


@triton.jit
def _reduce_instance_stats_kernel(
    sum_ptr, sumsq_ptr, mean_ptr, inv_std_ptr,
    N, C, NUM_TILES, HW,
    stats_stride_n, stats_stride_c, stats_stride_t,
    eps,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * stats_stride_n + c * stats_stride_c

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for start in range(0, NUM_TILES, BLOCK_T):
        offs = start + tl.arange(0, BLOCK_T)
        mask = offs < NUM_TILES
        ptrs = base + offs * stats_stride_t
        s = tl.load(sum_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.load(sumsq_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(s, axis=0)
        sum_sq += tl.sum(ss, axis=0)

    hw_f = tl.cast(HW, tl.float32)
    mean = sum_val / hw_f
    var = sum_sq / hw_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(inv_std_ptr + pid, inv_std)


@triton.jit
def _instance_norm2d_apply_from_stats_kernel(
    x_ptr, mean_ptr, inv_std_ptr, y_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    scale,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * stride_n + c * stride_c
    HW = H * W

    mean = tl.load(mean_ptr + pid).to(tl.float32)
    inv_std = tl.load(inv_std_ptr + pid).to(tl.float32)
    fused_scale = tl.cast(scale, tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h_idx = offs // W
        w_idx = offs - h_idx * W
        ptrs = x_ptr + base + h_idx * stride_h + w_idx * stride_w
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std * fused_scale
        out_ptrs = y_ptr + base + h_idx * stride_h + w_idx * stride_w
        tl.store(out_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


def conv2d_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.device.type == "xpu"
    assert weight.device.type == "xpu"
    assert bias.device.type == "xpu"
    assert x.ndim == 4 and weight.ndim == 4 and bias.ndim == 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, Cin, H, W = x.shape
    Cout, Cin_w, K, K_w = weight.shape
    assert Cin == Cin_w and K == 3 and K_w == 3

    stride_h = 1
    stride_w = 1
    pad_h = 0
    pad_w = 0
    dil_h = 1
    dil_w = 1

    H_out = (H + 2 * pad_h - dil_h * (K - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K - 1) - 1) // stride_w + 1

    y = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw = weight.stride()
    y_stride_n, y_stride_c, y_stride_h, y_stride_w = y.stride()

    BLOCK_CO = 32
    BLOCK_H = 8
    BLOCK_W = 32

    grid = (
        triton.cdiv(W_out, BLOCK_W),
        N * triton.cdiv(H_out, BLOCK_H),
        triton.cdiv(Cout, BLOCK_CO),
    )

    _conv2d_nchw_bias_kernel[grid](
        x, weight, bias, y,
        N, Cin, Cout, H, W, H_out, W_out,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        DIL_H=dil_h, DIL_W=dil_w,
        K=K,
        BLOCK_CO=BLOCK_CO, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=8,
        num_stages=2,
    )
    return y


def conv2d_bias_with_stats(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.device.type == "xpu"
    assert weight.device.type == "xpu"
    assert bias.device.type == "xpu"
    assert x.ndim == 4 and weight.ndim == 4 and bias.ndim == 1

    x_xpu = x.contiguous()
    w_xpu = weight.contiguous()
    b_xpu = bias.contiguous()

    N, Cin, H, W = x_xpu.shape
    Cout, Cin_w, K, K_w = w_xpu.shape
    assert Cin == Cin_w and K == 3 and K_w == 3

    stride_h = 1
    stride_w = 1
    pad_h = 0
    pad_w = 0
    dil_h = 1
    dil_w = 1

    H_out = (H + 2 * pad_h - dil_h * (K - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K - 1) - 1) // stride_w + 1

    y = torch.empty((N, Cout, H_out, W_out), device=x_xpu.device, dtype=x_xpu.dtype)

    BLOCK_CO = 32
    BLOCK_H = 8
    BLOCK_W = 32

    num_h_tiles = triton.cdiv(H_out, BLOCK_H)
    num_w_tiles = triton.cdiv(W_out, BLOCK_W)
    num_tiles = num_h_tiles * num_w_tiles

    partial_sum = torch.empty((N, Cout, num_tiles), device=x_xpu.device, dtype=torch.float32)
    partial_sumsq = torch.empty((N, Cout, num_tiles), device=x_xpu.device, dtype=torch.float32)

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x_xpu.stride()
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw = w_xpu.stride()
    y_stride_n, y_stride_c, y_stride_h, y_stride_w = y.stride()
    stats_stride_n, stats_stride_c, stats_stride_t = partial_sum.stride()

    grid = (
        num_w_tiles,
        N * num_h_tiles,
        triton.cdiv(Cout, BLOCK_CO),
    )

    _conv2d_nchw_bias_stats_kernel[grid](
        x_xpu, w_xpu, b_xpu, y, partial_sum, partial_sumsq,
        N, Cin, Cout, H, W, H_out, W_out,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        stats_stride_n, stats_stride_c, stats_stride_t,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        DIL_H=dil_h, DIL_W=dil_w,
        K=K,
        BLOCK_CO=BLOCK_CO, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=8,
        num_stages=1,
    )
    return y, partial_sum, partial_sumsq


def instancenorm_scale(x: torch.Tensor, scale: float, eps=1e-5):
    assert x.device.type == "xpu"
    assert x.dtype == torch.float16
    assert x.ndim == 4

    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    BLOCK_HW = 256
    grid = (N * C,)

    _instance_norm2d_scale_kernel[grid](
        x, y,
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        eps, scale,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
        num_stages=2,
    )
    return y


def instancenorm_scale_from_stats(
    x: torch.Tensor,
    partial_sum: torch.Tensor,
    partial_sumsq: torch.Tensor,
    scale: float,
    eps=1e-5,
):
    assert x.device.type == "xpu"
    assert partial_sum.device.type == "xpu"
    assert partial_sumsq.device.type == "xpu"
    assert x.dtype == torch.float16
    assert partial_sum.dtype == torch.float32
    assert partial_sumsq.dtype == torch.float32

    x_xpu = x.contiguous()
    N, C, H, W = x_xpu.shape
    y = torch.empty_like(x_xpu)

    NUM_TILES = partial_sum.shape[2]
    HW = H * W

    mean = torch.empty((N, C), device=x_xpu.device, dtype=torch.float32)
    inv_std = torch.empty((N, C), device=x_xpu.device, dtype=torch.float32)

    stats_stride_n, stats_stride_c, stats_stride_t = partial_sum.stride()

    _reduce_instance_stats_kernel[(N * C,)](
        partial_sum, partial_sumsq, mean, inv_std,
        N, C, NUM_TILES, HW,
        stats_stride_n, stats_stride_c, stats_stride_t,
        eps,
        BLOCK_T=256,
        num_warps=8,
        num_stages=1,
    )

    stride_n, stride_c, stride_h, stride_w = x_xpu.stride()
    _instance_norm2d_apply_from_stats_kernel[(N * C,)](
        x_xpu, mean, inv_std, y,
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        scale,
        BLOCK_HW=256,
        num_warps=8,
        num_stages=1,
    )
    return y


# ------------------------
# Fused top-level wrapper
# ------------------------
def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, inv_div_scale: float):
    assert x.device.type == "xpu"
    assert weight.device.type == "xpu"
    assert bias.device.type == "xpu"

    y, partial_sum, partial_sumsq = conv2d_bias_with_stats(x, weight, bias)
    out = instancenorm_scale_from_stats(y, partial_sum, partial_sumsq, inv_div_scale)
    return out


# --------------------------------------
# Original Model and input generators
# --------------------------------------
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.inv_div_scale = 1.0 / float(divide_by)

    def _ensure_xpu_params(self):
        if self.conv.weight.device.type != "xpu" or self.conv.weight.dtype != torch.float16:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
        else:
            self.conv.weight.data = self.conv.weight.data.contiguous()

        if self.conv.bias is not None:
            if self.conv.bias.device.type != "xpu" or self.conv.bias.dtype != torch.float16:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            else:
                self.conv.bias.data = self.conv.bias.data.contiguous()

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous()

        self._ensure_xpu_params()

        return kernel_function(
            x,
            self.conv.weight,
            self.conv.bias,
            self.inv_div_scale,
        )