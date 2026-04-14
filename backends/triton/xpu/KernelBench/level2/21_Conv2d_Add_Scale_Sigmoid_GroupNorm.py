# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sigmoid_exp2(x):
    log2e = 1.4426950408889634
    return 1.0 / (1.0 + tl.math.exp2(-x * log2e))


# -----------------------------------------------------------------------------
# Original Triton kernel kept for compatibility / fallback:
# fused Conv2D (3x3, stride=1, pad=0) + add bias + mul scale + sigmoid
# -----------------------------------------------------------------------------
@triton.jit
def _fused_conv_add_mul_sigmoid(
    x_ptr,
    w_ptr,
    bias_ptr,
    extra_bias_ptr,
    extra_scale_ptr,
    y_ptr,
    N, C_in, H, W,
    C_out, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_ebc, stride_esc,
    K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CO: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)

    M_total = N * OH * OW
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M_total

    HW = OH * OW
    n_idx = offs_m // HW
    rem = offs_m % HW
    ho = rem // OW
    wo = rem % OW

    co_start = pid_co * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < C_out

    acc = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    for ic in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                i_h = ho + kh
                i_w = wo + kw
                ptr_x = (
                    x_ptr
                    + n_idx * stride_xn
                    + ic * stride_xc
                    + i_h * stride_xh
                    + i_w * stride_xw
                )
                x_vals = tl.load(ptr_x, mask=mask_m, other=0.0).to(tl.float32)
                ptr_w = (
                    w_ptr
                    + offs_co * stride_wo
                    + ic * stride_wc
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_vals = tl.load(ptr_w, mask=mask_co, other=0.0).to(tl.float32)
                acc += x_vals[:, None] * w_vals[None, :]

    b = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    eb = tl.load(extra_bias_ptr + offs_co * stride_ebc, mask=mask_co, other=0.0).to(tl.float32)
    es = tl.load(extra_scale_ptr + offs_co * stride_esc, mask=mask_co, other=0.0).to(tl.float32)
    acc = (acc + eb[None, :]) * es[None, :]

    sig = _sigmoid_exp2(acc)

    ptr_y = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_co[None, :] * stride_yc
        + ho[:, None] * stride_yh
        + wo[:, None] * stride_yw
    )
    mask_out = mask_m[:, None] & mask_co[None, :]
    tl.store(ptr_y, sig.to(y_ptr.dtype.element_ty), mask=mask_out)


# -----------------------------------------------------------------------------
# Original Triton kernel kept for compatibility / fallback:
# GroupNorm NCHW with affine
# -----------------------------------------------------------------------------
@triton.jit
def _groupnorm_nchw_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N, C, H, W,
    G,
    stride_n, stride_c, stride_h, stride_w,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    group_size = C // G
    c0 = g * group_size
    HW = H * W
    base_n = n.to(tl.int64) * stride_n

    sum_x = tl.zeros([], dtype=tl.float32)
    sum_x2 = tl.zeros([], dtype=tl.float32)
    for ci in range(group_size):
        c_abs = c0 + ci
        ptr_c = x_ptr + base_n + c_abs * stride_c
        for offs in range(0, HW, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            m = idx < HW
            h = idx // W
            w = idx - h * W
            ptr = ptr_c + h * stride_h + w * stride_w
            vals = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
            sum_x += tl.sum(vals, axis=0)
            sum_x2 += tl.sum(vals * vals, axis=0)

    elems = group_size * HW
    inv_elems = 1.0 / elems
    mean = sum_x * inv_elems
    var = sum_x2 * inv_elems - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + eps)

    for ci in range(group_size):
        c_abs = c0 + ci
        in_ptr = x_ptr + base_n + c_abs * stride_c
        out_ptr = y_ptr + base_n + c_abs * stride_c
        gamma = tl.load(w_ptr + c_abs).to(tl.float32)
        beta = tl.load(b_ptr + c_abs).to(tl.float32)
        for offs in range(0, HW, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            m = idx < HW
            h = idx // W
            w = idx - h * W
            p_in = in_ptr + h * stride_h + w * stride_w
            p_out = out_ptr + h * stride_h + w * stride_w
            x_val = tl.load(p_in, mask=m, other=0.0).to(tl.float32)
            y_val = (x_val - mean) * inv_std
            y_val = y_val * gamma + beta
            tl.store(p_out, y_val.to(y_ptr.dtype.element_ty), mask=m)


# -----------------------------------------------------------------------------
# Original Triton post-op kernel kept for compatibility / fallback.
# -----------------------------------------------------------------------------
@triton.jit
def _pointwise_add_mul_sigmoid_nchw_tiled(
    x_ptr,
    extra_bias_ptr,
    extra_scale_ptr,
    y_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_ebc, stride_esc,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)
    pid_ncblk = tl.program_id(axis=1)

    n = pid_ncblk // tl.cdiv(C, BLOCK_C)
    c_blk = pid_ncblk % tl.cdiv(C, BLOCK_C)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < (H * W)
    mask_c = offs_c < C

    h = offs_hw // W
    w = offs_hw - h * W

    base_x = n.to(tl.int64) * stride_xn
    base_y = n.to(tl.int64) * stride_yn

    x_ptrs = (
        x_ptr
        + base_x
        + offs_c[:, None] * stride_xc
        + h[None, :] * stride_xh
        + w[None, :] * stride_xw
    )
    mask = mask_c[:, None] & mask_hw[None, :]
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    eb = tl.load(extra_bias_ptr + offs_c * stride_ebc, mask=mask_c, other=0.0).to(tl.float32)
    es = tl.load(extra_scale_ptr + offs_c * stride_esc, mask=mask_c, other=0.0).to(tl.float32)

    out = (x_vals + eb[:, None]) * es[:, None]
    out = _sigmoid_exp2(out)

    y_ptrs = (
        y_ptr
        + base_y
        + offs_c[:, None] * stride_yc
        + h[None, :] * stride_yh
        + w[None, :] * stride_yw
    )
    tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty), mask=mask)


# -----------------------------------------------------------------------------
# Stats pass: channel-vectorized within each GroupNorm group.
# XPU autotuned over HW tile and warps; grf_mode is passed as constexpr launch
# option (not inside triton.Config per XPU backend constraint).
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 4}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C': 4}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C': 4}, num_warps=32, num_stages=2),
    ],
    key=['H', 'W', 'G', 'C'],
)
@triton.jit
def _fused_stats_postop_groupnorm_nchw_kernel(
    x_ptr,
    extra_bias_ptr,
    extra_scale_ptr,
    sum_ptr,
    sumsq_ptr,
    N, C, H, W,
    G,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_ebc, stride_esc,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    group_size = C // G
    c0 = g * group_size
    HW = H * W
    base_n = n.to(tl.int64) * stride_xn

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < group_size
    c_abs = c0 + offs_c

    eb = tl.load(extra_bias_ptr + c_abs * stride_ebc, mask=mask_c, other=0.0).to(tl.float32)
    es = tl.load(extra_scale_ptr + c_abs * stride_esc, mask=mask_c, other=0.0).to(tl.float32)

    sum_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for offs in range(0, HW, BLOCK_HW):
        idx = offs + tl.arange(0, BLOCK_HW)
        mask_hw = idx < HW
        h = idx // W
        w = idx - h * W

        ptrs = (
            x_ptr
            + base_n
            + c_abs[:, None] * stride_xc
            + h[None, :] * stride_xh
            + w[None, :] * stride_xw
        )
        mask = mask_c[:, None] & mask_hw[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        vals = (vals + eb[:, None]) * es[:, None]
        vals = _sigmoid_exp2(vals)
        sum_x += tl.sum(vals, axis=1)
        sum_x2 += tl.sum(vals * vals, axis=1)

    total_sum = tl.sum(sum_x, axis=0)
    total_sum2 = tl.sum(sum_x2, axis=0)
    tl.store(sum_ptr + pid, total_sum)
    tl.store(sumsq_ptr + pid, total_sum2)


# -----------------------------------------------------------------------------
# Apply pass: channel-vectorized GroupNorm + fused post-op.
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 4}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C': 4}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C': 4}, num_warps=32, num_stages=2),
    ],
    key=['H', 'W', 'G', 'C'],
)
@triton.jit
def _fused_apply_postop_groupnorm_nchw_kernel(
    x_ptr,
    extra_bias_ptr,
    extra_scale_ptr,
    gn_w_ptr,
    gn_b_ptr,
    sum_ptr,
    sumsq_ptr,
    y_ptr,
    N, C, H, W,
    G,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_ebc, stride_esc,
    eps,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    group_size = C // G
    c0 = g * group_size
    HW = H * W

    stat_idx = n * G + g
    sum_x = tl.load(sum_ptr + stat_idx).to(tl.float32)
    sum_x2 = tl.load(sumsq_ptr + stat_idx).to(tl.float32)

    elems = group_size * HW
    inv_elems = 1.0 / elems
    mean = sum_x * inv_elems
    var = sum_x2 * inv_elems - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + eps)

    base_x = n.to(tl.int64) * stride_xn
    base_y = n.to(tl.int64) * stride_yn

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < group_size
    c_abs = c0 + offs_c

    eb = tl.load(extra_bias_ptr + c_abs * stride_ebc, mask=mask_c, other=0.0).to(tl.float32)
    es = tl.load(extra_scale_ptr + c_abs * stride_esc, mask=mask_c, other=0.0).to(tl.float32)
    gamma = tl.load(gn_w_ptr + c_abs, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(gn_b_ptr + c_abs, mask=mask_c, other=0.0).to(tl.float32)

    for offs in range(0, HW, BLOCK_HW):
        idx = offs + tl.arange(0, BLOCK_HW)
        mask_hw = idx < HW
        h = idx // W
        w = idx - h * W

        mask = mask_c[:, None] & mask_hw[None, :]

        p_in = (
            x_ptr
            + base_x
            + c_abs[:, None] * stride_xc
            + h[None, :] * stride_xh
            + w[None, :] * stride_xw
        )
        x_val = tl.load(p_in, mask=mask, other=0.0).to(tl.float32)
        x_val = (x_val + eb[:, None]) * es[:, None]
        x_val = _sigmoid_exp2(x_val)
        y_val = (x_val - mean) * inv_std
        y_val = y_val * gamma[:, None] + beta[:, None]

        p_out = (
            y_ptr
            + base_y
            + c_abs[:, None] * stride_yc
            + h[None, :] * stride_yh
            + w[None, :] * stride_yw
        )
        tl.store(p_out, y_val.to(y_ptr.dtype.element_ty), mask=mask)


def _to_xpu_contiguous(t, dtype):
    if t.device.type != "xpu" or t.dtype != dtype:
        t = t.to("xpu", dtype=dtype)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def kernel_function(
    x,
    conv_weight,
    conv_bias,
    extra_bias,
    extra_scale,
    gn_weight,
    gn_bias,
    num_groups,
    eps=1e-5
):
    x_xpu = _to_xpu_contiguous(x, torch.float16)
    conv_weight_xpu = _to_xpu_contiguous(conv_weight, torch.float16)
    conv_bias_xpu = _to_xpu_contiguous(conv_bias, torch.float16)
    extra_bias_xpu = _to_xpu_contiguous(extra_bias, torch.float16)
    extra_scale_xpu = _to_xpu_contiguous(extra_scale, torch.float16)
    gn_weight_xpu = _to_xpu_contiguous(gn_weight, torch.float16)
    gn_bias_xpu = _to_xpu_contiguous(gn_bias, torch.float16)

    y_conv = F.conv2d(x_xpu, conv_weight_xpu, conv_bias_xpu, stride=1, padding=0)

    eb_flat = extra_bias_xpu.reshape(-1)
    es_flat = extra_scale_xpu.reshape(-1)

    N2, C2, H2, W2 = y_conv.shape
    G = int(num_groups)

    stats = torch.empty((N2 * G,), device=y_conv.device, dtype=torch.float32)
    stats_sq = torch.empty((N2 * G,), device=y_conv.device, dtype=torch.float32)
    out = torch.empty_like(y_conv)

    grid = (N2 * G,)

    _fused_stats_postop_groupnorm_nchw_kernel[grid](
        y_conv,
        eb_flat,
        es_flat,
        stats,
        stats_sq,
        N2, C2, H2, W2,
        G,
        y_conv.stride(0), y_conv.stride(1), y_conv.stride(2), y_conv.stride(3),
        eb_flat.stride(0), es_flat.stride(0),
        grf_mode='auto',
    )

    _fused_apply_postop_groupnorm_nchw_kernel[grid](
        y_conv,
        eb_flat,
        es_flat,
        gn_weight_xpu,
        gn_bias_xpu,
        stats,
        stats_sq,
        out,
        N2, C2, H2, W2,
        G,
        y_conv.stride(0), y_conv.stride(1), y_conv.stride(2), y_conv.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        eb_flat.stride(0), es_flat.stride(0),
        float(eps),
        grf_mode='auto',
    )

    return out


batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.zeros(scale_shape))
        self.scale = nn.Parameter(torch.ones(out_channels, 1, 1))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self._xpu_prepared = False
        self._cached_bias_flat = None
        self._cached_scale_flat = None

    def _prepare_for_xpu(self):
        if not self._xpu_prepared:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.scale.data = self.scale.data.to("xpu", dtype=torch.float16).contiguous()
            self.group_norm.weight.data = self.group_norm.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.group_norm.bias.data = self.group_norm.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self._cached_bias_flat = self.bias.reshape(-1)
            self._cached_scale_flat = self.scale.reshape(-1)
            self._xpu_prepared = True

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

        self._prepare_for_xpu()

        return kernel_function(
            x,
            self.conv.weight,
            self.conv.bias,
            self._cached_bias_flat,
            self._cached_scale_flat,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
        )
