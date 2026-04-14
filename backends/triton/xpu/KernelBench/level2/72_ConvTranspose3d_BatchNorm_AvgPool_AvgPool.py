# ruff: noqa: E731
import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _conv_transpose3d_bn_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    bnw_ptr,
    bnb_ptr,
    mu_ptr,
    var_ptr,
    N,
    C_OUT,
    D_IN,
    H_IN,
    W_IN,
    D_OUT,
    H_OUT,
    W_OUT,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    x_sN,
    x_sC,
    x_sD,
    x_sH,
    x_sW,
    w_sCin,
    w_sCout,
    w_sKd,
    w_sKh,
    w_sKw,
    y_sN,
    y_sC,
    y_sD,
    y_sH,
    y_sW,
    eps,
    CIN: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    total = N * C_OUT * D_OUT * H_OUT * W_OUT
    mask = offsets < total

    ow = offsets % W_OUT
    tmp = offsets // W_OUT
    oh = tmp % H_OUT
    tmp = tmp // H_OUT
    od = tmp % D_OUT
    tmp = tmp // D_OUT
    co = tmp % C_OUT
    n = tmp // C_OUT

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for ic in range(CIN):
        for kd in range(KD):
            t_d = od + pad_d - kd * dil_d
            valid_d = (t_d >= 0) & (t_d % stride_d == 0)
            id_ = t_d // stride_d
            valid_d = valid_d & (id_ < D_IN)
            for kh in range(KH):
                t_h = oh + pad_h - kh * dil_h
                valid_h = (t_h >= 0) & (t_h % stride_h == 0)
                ih = t_h // stride_h
                valid_h = valid_h & (ih < H_IN)
                for kw in range(KW):
                    t_w = ow + pad_w - kw * dil_w
                    valid_w = (t_w >= 0) & (t_w % stride_w == 0)
                    iw = t_w // stride_w
                    valid_w = valid_w & (iw < W_IN)
                    vmask = mask & valid_d & valid_h & valid_w

                    x_index = n * x_sN + ic * x_sC + id_ * x_sD + ih * x_sH + iw * x_sW
                    x_vals = tl.load(x_ptr + x_index, mask=vmask, other=0.0)

                    w_index = ic * w_sCin + co * w_sCout + kd * w_sKd + kh * w_sKh + kw * w_sKw
                    w_vals = tl.load(w_ptr + w_index, mask=mask, other=0.0)

                    acc += x_vals * w_vals

    b_vals = tl.load(b_ptr + co, mask=mask, other=0.0)
    acc = acc + b_vals

    gamma = tl.load(bnw_ptr + co, mask=mask, other=0.0)
    beta = tl.load(bnb_ptr + co, mask=mask, other=0.0)
    mu = tl.load(mu_ptr + co, mask=mask, other=0.0)
    var = tl.load(var_ptr + co, mask=mask, other=0.0)
    rsigma = tl.sqrt(var + eps)
    scale = gamma / rsigma
    shift = beta - mu * scale
    out = acc * scale + shift

    y_index = n * y_sN + co * y_sC + od * y_sD + oh * y_sH + ow * y_sW
    tl.store(y_ptr + y_index, out, mask=mask)


@triton.jit
def _avgpool3d_fused_two_passes_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    sN_in,
    sC_in,
    sD_in,
    sH_in,
    sW_in,
    sN_out,
    sC_out,
    sD_out,
    sH_out,
    sW_out,
    BLOCK_SIZE: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    T = N * C * OD * OH * OW
    mask = offs < T

    spatial = OD * OH * OW
    nc = offs // spatial
    rem = offs - nc * spatial
    ohow = OH * OW
    od = rem // ohow
    rem2 = rem - od * ohow
    oh = rem2 // OW
    ow = rem2 - oh * OW

    n = nc // C
    c = nc - n * C

    d0 = od * S_D
    h0 = oh * S_H
    w0 = ow * S_W

    base = x_ptr + n * sN_in + c * sC_in + d0 * sD_in + h0 * sH_in + w0 * sW_in
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for kd in range(K_D):
        for kh in range(K_H):
            for kw in range(K_W):
                ptrs = base + kd * sD_in + kh * sH_in + kw * sW_in
                vals = tl.load(ptrs, mask=mask, other=0.0)
                acc += vals.to(tl.float32)
    scale = 1.0 / float(K_D * K_H * K_W)
    out_vals = (acc * scale).to(y_ptr.dtype.element_ty)

    out_ptrs = y_ptr + n * sN_out + c * sC_out + od * sD_out + oh * sH_out + ow * sW_out
    tl.store(out_ptrs, out_vals, mask=mask)


@triton.jit
def _direct_pooled_convtranspose3d_bn_parity_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    bnw_ptr,
    bnb_ptr,
    mu_ptr,
    var_ptr,
    N,
    C_OUT,
    D_IN,
    H_IN,
    W_IN,
    OD,
    OH,
    OW,
    x_sN,
    x_sC,
    x_sD,
    x_sH,
    x_sW,
    w_sCin,
    w_sCout,
    w_sKd,
    w_sKh,
    w_sKw,
    y_sN,
    y_sC,
    y_sD,
    y_sH,
    y_sW,
    eps,
    CIN: tl.constexpr,
    BLOCK: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C_OUT * OD * OH * OW
    mask = offs < total

    ow2 = offs % OW
    t0 = offs // OW
    oh2 = t0 % OH
    t1 = t0 // OH
    od2 = t1 % OD
    t2 = t1 // OD
    co = t2 % C_OUT
    n = t2 // C_OUT

    gamma = tl.load(bnw_ptr + co, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(bnb_ptr + co, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(mu_ptr + co, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + co, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + co, mask=mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mu * scale

    d_base = od2 * 2
    h_base = oh2 * 2
    w_base = ow2 * 2

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for ic in range(CIN):
        w000 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 0 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w001 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 0 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w002 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 0 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w010 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 1 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w011 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 1 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w012 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 1 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w020 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 2 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w021 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 2 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w022 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 0 * w_sKd + 2 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)

        w100 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 0 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w101 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 0 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w102 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 0 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w110 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 1 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w111 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 1 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w112 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 1 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w120 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 2 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w121 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 2 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w122 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 1 * w_sKd + 2 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)

        w200 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 0 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w201 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 0 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w202 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 0 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w210 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 1 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w211 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 1 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w212 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 1 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w220 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 2 * w_sKh + 0 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w221 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 2 * w_sKh + 1 * w_sKw, mask=mask, other=0.0).to(tl.float32)
        w222 = tl.load(w_ptr + ic * w_sCin + co * w_sCout + 2 * w_sKd + 2 * w_sKh + 2 * w_sKw, mask=mask, other=0.0).to(tl.float32)

        g000 = w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111
        g001 = w001 + w002 + w011 + w012 + w101 + w102 + w111 + w112
        g010 = w010 + w011 + w020 + w021 + w110 + w111 + w120 + w121
        g011 = w011 + w012 + w021 + w022 + w111 + w112 + w121 + w122
        g100 = w100 + w101 + w110 + w111 + w200 + w201 + w210 + w211
        g101 = w101 + w102 + w111 + w112 + w201 + w202 + w211 + w212
        g110 = w110 + w111 + w120 + w121 + w210 + w211 + w220 + w221
        g111 = w111 + w112 + w121 + w122 + w211 + w212 + w221 + w222

        for dd in range(2):
            id_ = d_base + dd
            valid_d = id_ < D_IN
            for hh in range(2):
                ih = h_base + hh
                valid_h = ih < H_IN
                for ww in range(2):
                    iw = w_base + ww
                    valid_w = iw < W_IN
                    vmask = mask & valid_d & valid_h & valid_w
                    x_idx = n * x_sN + ic * x_sC + id_ * x_sD + ih * x_sH + iw * x_sW
                    x_val = tl.load(x_ptr + x_idx, mask=vmask, other=0.0).to(tl.float32)

                    if dd == 0 and hh == 0 and ww == 0:
                        g = g000
                    elif dd == 0 and hh == 0 and ww == 1:
                        g = g001
                    elif dd == 0 and hh == 1 and ww == 0:
                        g = g010
                    elif dd == 0 and hh == 1 and ww == 1:
                        g = g011
                    elif dd == 1 and hh == 0 and ww == 0:
                        g = g100
                    elif dd == 1 and hh == 0 and ww == 1:
                        g = g101
                    elif dd == 1 and hh == 1 and ww == 0:
                        g = g110
                    else:
                        g = g111
                    acc += x_val * g

    pooled_conv = acc * (1.0 / 64.0) + bias
    out = pooled_conv * scale + shift

    y_idx = n * y_sN + co * y_sC + od2 * y_sD + oh2 * y_sH + ow2 * y_sW
    tl.store(y_ptr + y_idx, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _direct_pooled_convtranspose3d_bn_reduced_kernel(
    x_ptr,
    wg_ptr,
    bias_ptr,
    y_ptr,
    bn_scale_ptr,
    bn_shift_ptr,
    N,
    C_OUT,
    D_IN,
    H_IN,
    W_IN,
    OD,
    OH,
    OW,
    x_sN,
    x_sC,
    x_sD,
    x_sH,
    x_sW,
    wg_sCin,
    wg_sCout,
    wg_sGd,
    wg_sGh,
    wg_sGw,
    y_sN,
    y_sC,
    y_sD,
    y_sH,
    y_sW,
    CIN: tl.constexpr,
    BLOCK: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C_OUT * OD * OH * OW
    mask = offs < total

    ow2 = offs % OW
    t0 = offs // OW
    oh2 = t0 % OH
    t1 = t0 // OH
    od2 = t1 % OD
    t2 = t1 // OD
    co = t2 % C_OUT
    n = t2 // C_OUT

    scale = tl.load(bn_scale_ptr + co, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(bn_shift_ptr + co, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + co, mask=mask, other=0.0).to(tl.float32)

    d_base = od2 * 2
    h_base = oh2 * 2
    w_base = ow2 * 2

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for ic in tl.static_range(0, CIN):
        g_base = wg_ptr + ic * wg_sCin + co * wg_sCout

        for dd in tl.static_range(0, 2):
            id_ = d_base + dd
            valid_d = id_ < D_IN
            for hh in tl.static_range(0, 2):
                ih = h_base + hh
                valid_h = ih < H_IN
                for ww in tl.static_range(0, 2):
                    iw = w_base + ww
                    valid_w = iw < W_IN
                    vmask = mask & valid_d & valid_h & valid_w

                    x_idx = n * x_sN + ic * x_sC + id_ * x_sD + ih * x_sH + iw * x_sW
                    x_val = tl.load(x_ptr + x_idx, mask=vmask, other=0.0).to(tl.float32)

                    g = tl.load(
                        g_base + dd * wg_sGd + hh * wg_sGh + ww * wg_sGw,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)

                    acc += x_val * g

    pooled_conv = acc * (1.0 / 64.0) + bias
    out = pooled_conv * scale + shift

    y_idx = n * y_sN + co * y_sC + od2 * y_sD + oh2 * y_sH + ow2 * y_sW
    tl.store(y_ptr + y_idx, out.to(y_ptr.dtype.element_ty), mask=mask)


def _precompute_reduced_weights_and_bn(
    weight_xpu,
    bn_weight_xpu,
    bn_bias_xpu,
    bn_running_mean_xpu,
    bn_running_var_xpu,
    eps,
):
    w = weight_xpu.to(torch.float32)

    g000 = w[:, :, 0, 0, 0] + w[:, :, 0, 0, 1] + w[:, :, 0, 1, 0] + w[:, :, 0, 1, 1] + w[:, :, 1, 0, 0] + w[:, :, 1, 0, 1] + w[:, :, 1, 1, 0] + w[:, :, 1, 1, 1]
    g001 = w[:, :, 0, 0, 1] + w[:, :, 0, 0, 2] + w[:, :, 0, 1, 1] + w[:, :, 0, 1, 2] + w[:, :, 1, 0, 1] + w[:, :, 1, 0, 2] + w[:, :, 1, 1, 1] + w[:, :, 1, 1, 2]
    g010 = w[:, :, 0, 1, 0] + w[:, :, 0, 1, 1] + w[:, :, 0, 2, 0] + w[:, :, 0, 2, 1] + w[:, :, 1, 1, 0] + w[:, :, 1, 1, 1] + w[:, :, 1, 2, 0] + w[:, :, 1, 2, 1]
    g011 = w[:, :, 0, 1, 1] + w[:, :, 0, 1, 2] + w[:, :, 0, 2, 1] + w[:, :, 0, 2, 2] + w[:, :, 1, 1, 1] + w[:, :, 1, 1, 2] + w[:, :, 1, 2, 1] + w[:, :, 1, 2, 2]
    g100 = w[:, :, 1, 0, 0] + w[:, :, 1, 0, 1] + w[:, :, 1, 1, 0] + w[:, :, 1, 1, 1] + w[:, :, 2, 0, 0] + w[:, :, 2, 0, 1] + w[:, :, 2, 1, 0] + w[:, :, 2, 1, 1]
    g101 = w[:, :, 1, 0, 1] + w[:, :, 1, 0, 2] + w[:, :, 1, 1, 1] + w[:, :, 1, 1, 2] + w[:, :, 2, 0, 1] + w[:, :, 2, 0, 2] + w[:, :, 2, 1, 1] + w[:, :, 2, 1, 2]
    g110 = w[:, :, 1, 1, 0] + w[:, :, 1, 1, 1] + w[:, :, 1, 2, 0] + w[:, :, 1, 2, 1] + w[:, :, 2, 1, 0] + w[:, :, 2, 1, 1] + w[:, :, 2, 2, 0] + w[:, :, 2, 2, 1]
    g111 = w[:, :, 1, 1, 1] + w[:, :, 1, 1, 2] + w[:, :, 1, 2, 1] + w[:, :, 1, 2, 2] + w[:, :, 2, 1, 1] + w[:, :, 2, 1, 2] + w[:, :, 2, 2, 1] + w[:, :, 2, 2, 2]

    wg = torch.empty((w.shape[0], w.shape[1], 2, 2, 2), device=w.device, dtype=torch.float32)
    wg[:, :, 0, 0, 0] = g000
    wg[:, :, 0, 0, 1] = g001
    wg[:, :, 0, 1, 0] = g010
    wg[:, :, 0, 1, 1] = g011
    wg[:, :, 1, 0, 0] = g100
    wg[:, :, 1, 0, 1] = g101
    wg[:, :, 1, 1, 0] = g110
    wg[:, :, 1, 1, 1] = g111

    inv_std = torch.rsqrt(bn_running_var_xpu + eps)
    bn_scale = (bn_weight_xpu * inv_std).contiguous()
    bn_shift = (bn_bias_xpu - bn_running_mean_xpu * bn_scale).contiguous()

    return wg.contiguous(), bn_scale, bn_shift


def kernel_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    eps: float = 1e-5,
):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available")

    x_xpu = x.to("xpu", dtype=torch.float16).contiguous() if (x.device.type != "xpu" or x.dtype != torch.float16) else x.contiguous()
    weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous() if (weight.device.type != "xpu" or weight.dtype != torch.float16) else weight.contiguous()
    bias_xpu = bias.to("xpu", dtype=torch.float32).contiguous() if (bias.device.type != "xpu" or bias.dtype != torch.float32) else bias.contiguous()
    bn_weight_xpu = bn_weight.to("xpu", dtype=torch.float32).contiguous() if (bn_weight.device.type != "xpu" or bn_weight.dtype != torch.float32) else bn_weight.contiguous()
    bn_bias_xpu = bn_bias.to("xpu", dtype=torch.float32).contiguous() if (bn_bias.device.type != "xpu" or bn_bias.dtype != torch.float32) else bn_bias.contiguous()
    bn_running_mean_xpu = bn_running_mean.to("xpu", dtype=torch.float32).contiguous() if (bn_running_mean.device.type != "xpu" or bn_running_mean.dtype != torch.float32) else bn_running_mean.contiguous()
    bn_running_var_xpu = bn_running_var.to("xpu", dtype=torch.float32).contiguous() if (bn_running_var.device.type != "xpu" or bn_running_var.dtype != torch.float32) else bn_running_var.contiguous()

    N, C_in, D_in, H_in, W_in = x_xpu.shape
    Cin_w, C_out, Kd, Kh, Kw = weight_xpu.shape
    assert Cin_w == C_in and bias_xpu.numel() == C_out
    assert Kd == 3 and Kh == 3 and Kw == 3

    D_out = (D_in - 1) * 2 - 2 + (Kd - 1) + 1
    H_out = (H_in - 1) * 2 - 2 + (Kh - 1) + 1
    W_out = (W_in - 1) * 2 - 2 + (Kw - 1) + 1

    OD = (D_out - 4) // 4 + 1
    OH = (H_out - 4) // 4 + 1
    OW = (W_out - 4) // 4 + 1

    y = torch.empty((N, C_out, OD, OH, OW), device="xpu", dtype=torch.float16)

    wg_xpu, bn_scale_xpu, bn_shift_xpu = _precompute_reduced_weights_and_bn(
        weight_xpu, bn_weight_xpu, bn_bias_xpu, bn_running_mean_xpu, bn_running_var_xpu, eps
    )

    x_sN, x_sC, x_sD, x_sH, x_sW = x_xpu.stride()
    wg_sCin, wg_sCout, wg_sGd, wg_sGh, wg_sGw = wg_xpu.stride()
    y_sN, y_sC, y_sD, y_sH, y_sW = y.stride()

    BLOCK = 128
    total = N * C_out * OD * OH * OW
    grid = lambda META: (triton.cdiv(total, META["BLOCK"]),)

    _direct_pooled_convtranspose3d_bn_reduced_kernel[grid](
        x_xpu,
        wg_xpu,
        bias_xpu,
        y,
        bn_scale_xpu,
        bn_shift_xpu,
        N,
        C_out,
        D_in,
        H_in,
        W_in,
        OD,
        OH,
        OW,
        x_sN,
        x_sC,
        x_sD,
        x_sH,
        x_sW,
        wg_sCin,
        wg_sCout,
        wg_sGd,
        wg_sGh,
        wg_sGw,
        y_sN,
        y_sC,
        y_sD,
        y_sH,
        y_sW,
        CIN=C_in,
        BLOCK=BLOCK,
        grf_mode="auto",
        num_warps=8,
        num_stages=1,
    )

    return y


batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.bias_shape = bias_shape
        self._moved_to_xpu = False

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        if not self._moved_to_xpu:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float32).contiguous()
            self.bn.weight.data = self.bn.weight.data.to("xpu", dtype=torch.float32).contiguous()
            self.bn.bias.data = self.bn.bias.data.to("xpu", dtype=torch.float32).contiguous()
            self.bn.running_mean.data = self.bn.running_mean.data.to("xpu", dtype=torch.float32).contiguous()
            self.bn.running_var.data = self.bn.running_var.data.to("xpu", dtype=torch.float32).contiguous()
            self._moved_to_xpu = True

        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
        )