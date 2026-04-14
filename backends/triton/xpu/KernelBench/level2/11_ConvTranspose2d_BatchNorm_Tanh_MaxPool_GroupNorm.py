# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _conv_transpose_autotune_configs():
    configs = [
        triton.Config({"BLOCK_M": 32, "BLOCK_CO": 32, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_CO": 64, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_CO": 32, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_CO": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_CO": 128, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_CO": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_CO": 128, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_CO": 64, "GROUP_SIZE_M": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_CO": 128, "GROUP_SIZE_M": 2}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_CO": 128, "GROUP_SIZE_M": 4}, num_warps=16, num_stages=3),
        # Required large-tile / high-warp Intel XPU candidate
        triton.Config({"BLOCK_M": 256, "BLOCK_CO": 256, "GROUP_SIZE_M": 1}, num_warps=32, num_stages=2),
    ]
    return configs


def _maxpool_autotune_configs():
    configs = [
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 256}, num_warps=32, num_stages=2),
    ]
    return configs


def _groupnorm_autotune_configs():
    configs = [
        triton.Config({"BLOCK_W": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=2),
    ]
    return configs


@triton.autotune(
    configs=_conv_transpose_autotune_configs(),
    key=["N", "C_IN", "C_OUT", "H_IN", "W_IN", "H_OUT", "W_OUT"],
)
@triton.jit
def _conv_transpose2d_bn_tanh_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    y_ptr,
    N,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    STRIDE_XN,
    STRIDE_XC,
    STRIDE_XH,
    STRIDE_XW,
    STRIDE_WCI,
    STRIDE_WCO,
    STRIDE_WKH,
    STRIDE_WKW,
    STRIDE_YN,
    STRIDE_YC,
    STRIDE_YH,
    STRIDE_YW,
    PAD_H,
    PAD_W,
    EPS,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N * H_OUT * W_OUT, BLOCK_M)
    num_pid_co = tl.cdiv(C_OUT, BLOCK_CO)
    num_pid_in_group = GROUP_SIZE_M * num_pid_co
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_co = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (N * H_OUT * W_OUT)

    tmp0 = offs_m // W_OUT
    wo = offs_m % W_OUT
    ho = tmp0 % H_OUT
    n = tmp0 // H_OUT

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < C_OUT

    gamma = tl.load(gamma_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    b_conv = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)

    inv_std = tl.rsqrt(var + EPS)
    scale = gamma * inv_std
    shift = beta + (b_conv - mean) * scale

    acc = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    for ci in range(0, C_IN):
        base_x_nci = n * STRIDE_XN + ci * STRIDE_XC
        base_w_ci = w_ptr + ci * STRIDE_WCI
        for kh in range(0, K_H):
            hi = ho + PAD_H - kh
            valid_h = (hi >= 0) & (hi < H_IN)
            w_kh = base_w_ci + kh * STRIDE_WKH
            for kw in range(0, K_W):
                wi = wo + PAD_W - kw
                valid_w = (wi >= 0) & (wi < W_IN)
                m_mask = mask_m & valid_h & valid_w
                x_ptrs = x_ptr + base_x_nci + hi * STRIDE_XH + wi * STRIDE_XW
                x_vals = tl.load(x_ptrs, mask=m_mask, other=0.0).to(tl.float32)
                w_ptrs = w_kh + offs_co * STRIDE_WCO + kw * STRIDE_WKW
                w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0).to(tl.float32)
                acc += x_vals[:, None] * w_vals[None, :]

    y_tile = acc * scale[None, :] + shift[None, :]

    abs_y = tl.abs(y_tile)
    t = tl.exp(-2.0 * abs_y)
    tanh_pos = (1.0 - t) / (1.0 + t)
    sign = tl.where(y_tile >= 0, 1.0, -1.0)
    y_act = sign * tanh_pos

    y_ptrs = (
        y_ptr
        + n[:, None] * STRIDE_YN
        + offs_co[None, :] * STRIDE_YC
        + ho[:, None] * STRIDE_YH
        + wo[:, None] * STRIDE_YW
    )
    store_mask = mask_m[:, None] & mask_co[None, :]
    tl.store(y_ptrs, y_act.to(y_ptr.dtype.element_ty), mask=store_mask)


@triton.jit
def _fused_maxpool2d_groupnorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    GROUPS,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C // GROUPS
    c_base = g * group_size
    offs_c = tl.arange(0, BLOCK_C)
    c_idx = c_base + offs_c
    mask_c = offs_c < group_size

    gamma = tl.load(w_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)

    sum_vec = tl.zeros([BLOCK_C], dtype=tl.float32)
    sumsq_vec = tl.zeros([BLOCK_C], dtype=tl.float32)

    base_n = n * stride_xn
    base_c = base_n + c_idx * stride_xc

    for oh in tl.range(0, H_OUT):
        h0 = oh * 2
        h1 = h0 + 1
        h0_in = h0 < H
        h1_in = h1 < H
        h0_off = h0 * stride_xh
        h1_off = h1 * stride_xh
        for ow in tl.range(0, W_OUT):
            w0 = ow * 2
            w1 = w0 + 1
            w0_in = w0 < W
            w1_in = w1 < W
            w0_off = w0 * stride_xw
            w1_off = w1 * stride_xw

            ptr00 = x_ptr + base_c + h0_off + w0_off
            ptr01 = x_ptr + base_c + h0_off + w1_off
            ptr10 = x_ptr + base_c + h1_off + w0_off
            ptr11 = x_ptr + base_c + h1_off + w1_off

            m00 = mask_c & h0_in & w0_in
            m01 = mask_c & h0_in & w1_in
            m10 = mask_c & h1_in & w0_in
            m11 = mask_c & h1_in & w1_in

            v00 = tl.load(ptr00, mask=m00, other=-float("inf")).to(tl.float32)
            v01 = tl.load(ptr01, mask=m01, other=-float("inf")).to(tl.float32)
            v10 = tl.load(ptr10, mask=m10, other=-float("inf")).to(tl.float32)
            v11 = tl.load(ptr11, mask=m11, other=-float("inf")).to(tl.float32)

            vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
            sum_vec += vmax
            sumsq_vec += vmax * vmax

    total_sum = tl.sum(sum_vec, axis=0)
    total_sumsq = tl.sum(sumsq_vec, axis=0)

    elems = group_size * H_OUT * W_OUT
    inv_elems = 1.0 / elems
    mean = total_sum * inv_elems
    var = total_sumsq * inv_elems - mean * mean
    inv_std = tl.rsqrt(var + eps)

    base_ny = n * stride_yn
    base_cy = base_ny + c_idx * stride_yc

    for oh in tl.range(0, H_OUT):
        h0 = oh * 2
        h1 = h0 + 1
        h0_in = h0 < H
        h1_in = h1 < H
        h0_off = h0 * stride_xh
        h1_off = h1 * stride_xh
        for ow in tl.range(0, W_OUT):
            w0 = ow * 2
            w1 = w0 + 1
            w0_in = w0 < W
            w1_in = w1 < W
            w0_off = w0 * stride_xw
            w1_off = w1 * stride_xw

            ptr00 = x_ptr + base_c + h0_off + w0_off
            ptr01 = x_ptr + base_c + h0_off + w1_off
            ptr10 = x_ptr + base_c + h1_off + w0_off
            ptr11 = x_ptr + base_c + h1_off + w1_off

            m00 = mask_c & h0_in & w0_in
            m01 = mask_c & h0_in & w1_in
            m10 = mask_c & h1_in & w0_in
            m11 = mask_c & h1_in & w1_in

            v00 = tl.load(ptr00, mask=m00, other=-float("inf")).to(tl.float32)
            v01 = tl.load(ptr01, mask=m01, other=-float("inf")).to(tl.float32)
            v10 = tl.load(ptr10, mask=m10, other=-float("inf")).to(tl.float32)
            v11 = tl.load(ptr11, mask=m11, other=-float("inf")).to(tl.float32)

            vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
            out_vals = (vmax - mean) * inv_std
            out_vals = out_vals * gamma + beta
            out_ptrs = y_ptr + base_cy + oh * stride_yh + ow * stride_yw
            tl.store(out_ptrs, out_vals.to(y_ptr.dtype.element_ty), mask=mask_c)


@triton.autotune(
    configs=_maxpool_autotune_configs(),
    key=["N", "C", "H", "W", "H_OUT", "W_OUT"],
)
@triton.jit
def _maxpool2d_compact_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    total_hw = N * H_OUT * W_OUT
    mask_hw = offs_hw < total_hw

    tmp = offs_hw // W_OUT
    ow = offs_hw % W_OUT
    oh = tmp % H_OUT
    n = tmp // H_OUT

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    h0 = oh * 2
    h1 = h0 + 1
    w0 = ow * 2
    w1 = w0 + 1

    base_x = n[:, None] * stride_xn + offs_c[None, :] * stride_xc

    m00 = mask_hw[:, None] & mask_c[None, :] & (h0[:, None] < H) & (w0[:, None] < W)
    m01 = mask_hw[:, None] & mask_c[None, :] & (h0[:, None] < H) & (w1[:, None] < W)
    m10 = mask_hw[:, None] & mask_c[None, :] & (h1[:, None] < H) & (w0[:, None] < W)
    m11 = mask_hw[:, None] & mask_c[None, :] & (h1[:, None] < H) & (w1[:, None] < W)

    p00 = x_ptr + base_x + h0[:, None] * stride_xh + w0[:, None] * stride_xw
    p01 = x_ptr + base_x + h0[:, None] * stride_xh + w1[:, None] * stride_xw
    p10 = x_ptr + base_x + h1[:, None] * stride_xh + w0[:, None] * stride_xw
    p11 = x_ptr + base_x + h1[:, None] * stride_xh + w1[:, None] * stride_xw

    v00 = tl.load(p00, mask=m00, other=-float("inf")).to(tl.float32)
    v01 = tl.load(p01, mask=m01, other=-float("inf")).to(tl.float32)
    v10 = tl.load(p10, mask=m10, other=-float("inf")).to(tl.float32)
    v11 = tl.load(p11, mask=m11, other=-float("inf")).to(tl.float32)

    vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    out_ptrs = (
        y_ptr
        + n[:, None] * stride_yn
        + offs_c[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )
    tl.store(out_ptrs, vmax.to(y_ptr.dtype.element_ty), mask=mask_hw[:, None] & mask_c[None, :])


@triton.autotune(
    configs=_groupnorm_autotune_configs(),
    key=["N", "C", "H", "W", "GROUPS"],
)
@triton.jit
def _groupnorm_from_compact_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    GROUPS,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C // GROUPS
    c_base = g * group_size
    offs_c = tl.arange(0, BLOCK_C)
    c_idx = c_base + offs_c
    mask_c = offs_c < group_size

    gamma = tl.load(w_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)

    sum_vec = tl.zeros([BLOCK_C], dtype=tl.float32)
    sumsq_vec = tl.zeros([BLOCK_C], dtype=tl.float32)

    base_x = x_ptr + n * stride_xn + c_base * stride_xc
    base_y = y_ptr + n * stride_yn + c_base * stride_yc

    for oh in tl.range(0, H):
        row_x = base_x + oh * stride_xh
        for ow_blk in tl.range(0, W, BLOCK_W):
            x_bp = tl.make_block_ptr(
                base=row_x,
                shape=(group_size, W),
                strides=(stride_xc, stride_xw),
                offsets=(0, ow_blk),
                block_shape=(BLOCK_C, BLOCK_W),
                order=(1, 0),
            )
            vals = tl.load(x_bp, boundary_check=(0, 1)).to(tl.float32)
            sum_vec += tl.sum(vals, axis=1)
            sumsq_vec += tl.sum(vals * vals, axis=1)

    total_sum = tl.sum(sum_vec, axis=0)
    total_sumsq = tl.sum(sumsq_vec, axis=0)

    elems = group_size * H * W
    inv_elems = 1.0 / elems
    mean = total_sum * inv_elems
    var = total_sumsq * inv_elems - mean * mean
    inv_std = tl.rsqrt(var + eps)

    for oh in tl.range(0, H):
        row_x = base_x + oh * stride_xh
        row_y = base_y + oh * stride_yh
        for ow_blk in tl.range(0, W, BLOCK_W):
            x_bp = tl.make_block_ptr(
                base=row_x,
                shape=(group_size, W),
                strides=(stride_xc, stride_xw),
                offsets=(0, ow_blk),
                block_shape=(BLOCK_C, BLOCK_W),
                order=(1, 0),
            )
            y_bp = tl.make_block_ptr(
                base=row_y,
                shape=(group_size, W),
                strides=(stride_yc, stride_yw),
                offsets=(0, ow_blk),
                block_shape=(BLOCK_C, BLOCK_W),
                order=(1, 0),
            )
            vals = tl.load(x_bp, boundary_check=(0, 1)).to(tl.float32)
            out = (vals - mean) * inv_std
            out = out * gamma[:, None] + beta[:, None]
            tl.store(y_bp, out.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


def conv_transpose_bn_tanh(x, w_ct, b_ct, bn_weight, bn_bias, running_mean, running_var, eps):
    assert x.device.type == "xpu"
    N, C_in, H_in, W_in = x.shape
    Cin_w, C_out, kH, kW = w_ct.shape
    assert Cin_w == C_in and b_ct.shape[0] == C_out

    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1
    out_pad_h = 0
    out_pad_w = 0
    dil_h = 1
    dil_w = 1

    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    sxn, sxc, sxh, sxw = x.stride()
    swci, swco, swkh, swkw = w_ct.stride()
    syn, syc, syh, syw = y.stride()

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]) * triton.cdiv(C_out, META["BLOCK_CO"]),
    )

    _conv_transpose2d_bn_tanh_kernel[grid](
        x,
        w_ct,
        b_ct,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        y,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        sxn,
        sxc,
        sxh,
        sxw,
        swci,
        swco,
        swkh,
        swkw,
        syn,
        syc,
        syh,
        syw,
        pad_h,
        pad_w,
        float(eps),
        K_H=kH,
        K_W=kW,
    )
    return y


def maxpool_groupnorm(x, gn_weight, gn_bias):
    assert x.device.type == "xpu"
    N, C, H, W = x.shape
    GROUPS = 8
    assert C % GROUPS == 0

    KH = KW = 2
    SH = SW = 2
    H_OUT = (H - KH) // SH + 1
    W_OUT = (W - KW) // SW + 1

    pooled = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
    y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    sxn, sxc, sxh, sxw = x.stride()
    spn, spc, sph, spw = pooled.stride()
    syn, syc, syh, syw = y.stride()

    grid_pool = lambda META: (
        triton.cdiv(N * H_OUT * W_OUT, META["BLOCK_HW"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    _maxpool2d_compact_kernel[grid_pool](
        x,
        pooled,
        N,
        C,
        H,
        W,
        H_OUT,
        W_OUT,
        sxn,
        sxc,
        sxh,
        sxw,
        spn,
        spc,
        sph,
        spw,
    )

    group_size = C // GROUPS
    grid_gn = (N * GROUPS,)
    _groupnorm_from_compact_kernel[grid_gn](
        pooled,
        gn_weight,
        gn_bias,
        y,
        N,
        C,
        H_OUT,
        W_OUT,
        GROUPS,
        spn,
        spc,
        sph,
        spw,
        syn,
        syc,
        syh,
        syw,
        1e-5,
        BLOCK_C=group_size,
    )
    return y


def kernel_function(
    x,
    w_ct,
    b_ct,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    gn_weight,
    gn_bias,
    bn_eps=1e-5,
):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if w_ct.device.type != "xpu" or w_ct.dtype != torch.float16:
        w_ct_xpu = w_ct.to("xpu", dtype=torch.float16).contiguous()
    else:
        w_ct_xpu = w_ct.contiguous()

    if b_ct.device.type != "xpu":
        b_ct_xpu = b_ct.to("xpu").contiguous()
    else:
        b_ct_xpu = b_ct.contiguous()

    if bn_weight.device.type != "xpu":
        bn_weight_xpu = bn_weight.to("xpu").contiguous()
    else:
        bn_weight_xpu = bn_weight.contiguous()

    if bn_bias.device.type != "xpu":
        bn_bias_xpu = bn_bias.to("xpu").contiguous()
    else:
        bn_bias_xpu = bn_bias.contiguous()

    if running_mean.device.type != "xpu":
        running_mean_xpu = running_mean.to("xpu").contiguous()
    else:
        running_mean_xpu = running_mean.contiguous()

    if running_var.device.type != "xpu":
        running_var_xpu = running_var.to("xpu").contiguous()
    else:
        running_var_xpu = running_var.contiguous()

    if gn_weight.device.type != "xpu":
        gn_weight_xpu = gn_weight.to("xpu").contiguous()
    else:
        gn_weight_xpu = gn_weight.contiguous()

    if gn_bias.device.type != "xpu":
        gn_bias_xpu = gn_bias.to("xpu").contiguous()
    else:
        gn_bias_xpu = gn_bias.contiguous()

    y1 = conv_transpose_bn_tanh(
        x_xpu,
        w_ct_xpu,
        b_ct_xpu,
        bn_weight_xpu,
        bn_bias_xpu,
        running_mean_xpu,
        running_var_xpu,
        bn_eps,
    )
    y2 = maxpool_groupnorm(y1, gn_weight_xpu, gn_bias_xpu)
    return y2


batch_size = 512
in_channels = 64
out_channels = 128
kernel_size = 5
stride = 1
padding = 1
groups = 8
num_groups = 8
height = width = 32


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        if self.conv_transpose.weight.device.type != "xpu":
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
        else:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous()

        if self.conv_transpose.bias.device.type != "xpu":
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu").contiguous()
        else:
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.contiguous()

        if self.batch_norm.weight.device.type != "xpu":
            self.batch_norm.weight.data = self.batch_norm.weight.data.to("xpu").contiguous()
        else:
            self.batch_norm.weight.data = self.batch_norm.weight.data.contiguous()

        if self.batch_norm.bias.device.type != "xpu":
            self.batch_norm.bias.data = self.batch_norm.bias.data.to("xpu").contiguous()
        else:
            self.batch_norm.bias.data = self.batch_norm.bias.data.contiguous()

        if self.batch_norm.running_mean.device.type != "xpu":
            self.batch_norm.running_mean.data = self.batch_norm.running_mean.data.to("xpu").contiguous()
        else:
            self.batch_norm.running_mean.data = self.batch_norm.running_mean.data.contiguous()

        if self.batch_norm.running_var.device.type != "xpu":
            self.batch_norm.running_var.data = self.batch_norm.running_var.data.to("xpu").contiguous()
        else:
            self.batch_norm.running_var.data = self.batch_norm.running_var.data.contiguous()

        if self.group_norm.weight.device.type != "xpu":
            self.group_norm.weight.data = self.group_norm.weight.data.to("xpu").contiguous()
        else:
            self.group_norm.weight.data = self.group_norm.weight.data.contiguous()

        if self.group_norm.bias.device.type != "xpu":
            self.group_norm.bias.data = self.group_norm.bias.data.to("xpu").contiguous()
        else:
            self.group_norm.bias.data = self.group_norm.bias.data.contiguous()

        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.group_norm.weight,
            self.group_norm.bias,
        )