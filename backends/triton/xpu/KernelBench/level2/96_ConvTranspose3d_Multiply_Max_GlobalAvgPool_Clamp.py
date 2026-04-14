# ruff: noqa: E731
import sys
import math
import torch
import triton
import triton.language as tl
import torch.nn as nn


def _conv_transpose3d_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=32, num_stages=2),
    ]


def _maxpool3d_autotune_configs():
    return [
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=2),
    ]


def _avgpool3d_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=32, num_stages=2),
    ]


# -------------------------------------------------------------------
# Original kernel kept for compatibility.
# -------------------------------------------------------------------
@triton.jit
def _conv_transpose3d_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    scale_ptr,
    y_ptr,
    N, C_OUT,
    D, H, W,
    D_OUT, H_OUT, W_OUT,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_ci, w_stride_co, w_stride_kd, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    BLOCK_SIZE: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    C_IN: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_nc = tl.program_id(1)
    n = pid_nc // C_OUT
    co = pid_nc % C_OUT
    total_spatial = D_OUT * H_OUT * W_OUT
    offs = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = offs < total_spatial
    ow = offs % W_OUT
    tmp = offs // W_OUT
    oh = tmp % H_OUT
    od = tmp // H_OUT
    y_base = y_ptr + n * y_stride_n + co * y_stride_c
    x_n_base = x_ptr + n * x_stride_n
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bias_val = tl.load(b_ptr + co)
    scale_val = tl.load(scale_ptr)
    for ci in range(C_IN):
        w_ci_base = w_ptr + ci * w_stride_ci + co * w_stride_co
        for kd in range(KD):
            base_d = od + PAD - kd
            even_d = ((base_d & (STRIDE - 1)) == 0)
            idv = base_d // STRIDE
            valid_d = (idv >= 0) & (idv < D) & even_d & out_mask
            for kh in range(KH):
                base_h = oh + PAD - kh
                even_h = ((base_h & (STRIDE - 1)) == 0)
                ihv = base_h // STRIDE
                valid_dh = valid_d & (ihv >= 0) & (ihv < H) & even_h
                for kw in range(KW):
                    base_w = ow + PAD - kw
                    even_w = ((base_w & (STRIDE - 1)) == 0)
                    iwv = base_w // STRIDE
                    valid = valid_dh & (iwv >= 0) & (iwv < W) & even_w
                    x_ptrs = x_n_base + ci * x_stride_c + idv * x_stride_d + ihv * x_stride_h + iwv * x_stride_w
                    x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
                    w_val = tl.load(w_ci_base + kd * w_stride_kd + kh * w_stride_kh + kw * w_stride_kw)
                    acc += x_vals * w_val
    acc = (acc + bias_val) * scale_val
    y_ptrs = y_base + od * y_stride_d + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptrs, acc, mask=out_mask)


# -------------------------------------------------------------------
# Optimized conv-transpose kernel with autotune.
# grf_mode is a compiler option on XPU, so it is declared but not
# passed via triton.Config().
# -------------------------------------------------------------------
@triton.autotune(
    configs=_conv_transpose3d_autotune_configs(),
    key=["C_OUT", "D", "H", "W", "D_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def _conv_transpose3d_fused_kernel_specialized(
    x_ptr,
    w_ptr,
    b_ptr,
    scale_ptr,
    y_ptr,
    N, C_OUT,
    D, H, W,
    D_OUT, H_OUT, W_OUT,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_ci, w_stride_co, w_stride_kd, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    BLOCK_SIZE: tl.constexpr,
    C_IN: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_nc = tl.program_id(1)

    n = pid_nc // C_OUT
    co = pid_nc % C_OUT

    total_spatial = D_OUT * H_OUT * W_OUT
    offs = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = offs < total_spatial

    ow = offs % W_OUT
    tmp = offs // W_OUT
    oh = tmp % H_OUT
    od = tmp // H_OUT

    id_c = od // 2
    ih_c = oh // 2
    iw_c = ow // 2

    id_l = (od - 1) // 2
    ih_l = (oh - 1) // 2
    iw_l = (ow - 1) // 2

    valid_d_c = (id_c >= 0) & (id_c < D)
    valid_h_c = (ih_c >= 0) & (ih_c < H)
    valid_w_c = (iw_c >= 0) & (iw_c < W)

    valid_d_l = ((od & 1) != 0) & (id_l >= 0) & (id_l < D)
    valid_h_l = ((oh & 1) != 0) & (ih_l >= 0) & (ih_l < H)
    valid_w_l = ((ow & 1) != 0) & (iw_l >= 0) & (iw_l < W)

    y_base = y_ptr + n * y_stride_n + co * y_stride_c
    x_n_base = x_ptr + n * x_stride_n

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bias_val = tl.load(b_ptr + co).to(tl.float32)
    scale_val = tl.load(scale_ptr).to(tl.float32)

    for ci in range(C_IN):
        w_ci_base = w_ptr + ci * w_stride_ci + co * w_stride_co
        x_ci_base = x_n_base + ci * x_stride_c

        valid = out_mask & valid_d_c & valid_h_c & valid_w_c
        x_vals = tl.load(
            x_ci_base + id_c * x_stride_d + ih_c * x_stride_h + iw_c * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kd + w_stride_kh + w_stride_kw
        ).to(tl.float32)

        valid = out_mask & valid_d_l & valid_h_c & valid_w_c
        x_vals = tl.load(
            x_ci_base + id_l * x_stride_d + ih_c * x_stride_h + iw_c * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kh + w_stride_kw
        ).to(tl.float32)

        valid = out_mask & valid_d_c & valid_h_l & valid_w_c
        x_vals = tl.load(
            x_ci_base + id_c * x_stride_d + ih_l * x_stride_h + iw_c * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kd + w_stride_kw
        ).to(tl.float32)

        valid = out_mask & valid_d_c & valid_h_c & valid_w_l
        x_vals = tl.load(
            x_ci_base + id_c * x_stride_d + ih_c * x_stride_h + iw_l * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kd + w_stride_kh
        ).to(tl.float32)

        valid = out_mask & valid_d_l & valid_h_l & valid_w_c
        x_vals = tl.load(
            x_ci_base + id_l * x_stride_d + ih_l * x_stride_h + iw_c * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kw
        ).to(tl.float32)

        valid = out_mask & valid_d_l & valid_h_c & valid_w_l
        x_vals = tl.load(
            x_ci_base + id_l * x_stride_d + ih_c * x_stride_h + iw_l * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kh
        ).to(tl.float32)

        valid = out_mask & valid_d_c & valid_h_l & valid_w_l
        x_vals = tl.load(
            x_ci_base + id_c * x_stride_d + ih_l * x_stride_h + iw_l * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(
            w_ci_base + w_stride_kd
        ).to(tl.float32)

        valid = out_mask & valid_d_l & valid_h_l & valid_w_l
        x_vals = tl.load(
            x_ci_base + id_l * x_stride_d + ih_l * x_stride_h + iw_l * x_stride_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        acc += x_vals * tl.load(w_ci_base).to(tl.float32)

    acc = (acc + bias_val) * scale_val
    y_ptrs = y_base + od * y_stride_d + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


# -------------------------------------------------------------------
# Maxpool kernel with autotune.
# -------------------------------------------------------------------
@triton.autotune(
    configs=_maxpool3d_autotune_configs(),
    key=["C", "D", "H", "W", "OD", "OH", "OW"],
)
@triton.jit
def _max_pool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    KERNEL_D: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_D: tl.constexpr, DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    tmp = pid0
    oh = tmp % OH
    tmp //= OH
    od = tmp % OD
    tmp //= OD
    nc = tmp
    n = nc // C
    c = nc % C
    ow_start = pid1 * BLOCK_W
    ow = ow_start + tl.arange(0, BLOCK_W)
    ow_mask = ow < OW
    acc = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
    for kd in range(KERNEL_D):
        in_d = od * STRIDE_D - PAD_D + kd * DIL_D
        valid_d = (in_d >= 0) & (in_d < D)
        for kh in range(KERNEL_H):
            in_h = oh * STRIDE_H - PAD_H + kh * DIL_H
            valid_h = (in_h >= 0) & (in_h < H)
            base_ptr = x_ptr + n * stride_n + c * stride_c + in_d * stride_d + in_h * stride_h
            for kw in range(KERNEL_W):
                in_w = ow * STRIDE_W - PAD_W + kw * DIL_W
                valid_w = (in_w >= 0) & (in_w < W)
                mask = ow_mask & valid_d & valid_h & valid_w
                ptrs = base_ptr + in_w * stride_w
                vals = tl.load(ptrs, mask=mask, other=-float("inf"))
                acc = tl.maximum(acc, vals.to(tl.float32))
    y_base = y_ptr + n * y_stride_n + c * y_stride_c + od * y_stride_d + oh * y_stride_h
    y_ptrs = y_base + ow * y_stride_w
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=ow_mask)


# -------------------------------------------------------------------
# Avgpool+clamp kernel with autotune.
# -------------------------------------------------------------------
@triton.autotune(
    configs=_avgpool3d_autotune_configs(),
    key=["C", "D", "H", "W"],
)
@triton.jit
def _avgpool3d_clamp_ncdhw_1x1x1(
    x_ptr, y_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    valid_nc = (n < N) & (c < C)
    base = x_ptr + n * stride_n + c * stride_c
    HW = H * W
    DHW = D * HW
    acc = tl.zeros((), dtype=tl.float32)
    for start in tl.range(0, DHW, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = (idx < DHW) & valid_nc
        d = idx // HW
        rem = idx % HW
        h = rem // W
        w = rem % W
        ptrs = base + d * stride_d + h * stride_h + w * stride_w
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=0)
    mean = acc / tl.full((), DHW, dtype=tl.float32)
    mean = tl.maximum(mean, 0.0)
    mean = tl.minimum(mean, 1.0)
    y_ptrs = y_ptr + n * out_stride_n + c * out_stride_c
    tl.store(y_ptrs, mean.to(y_ptr.dtype.element_ty), mask=valid_nc)


def _conv_transpose3d_mul_scale(x, weight, bias, scale):
    if not (isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor)
            and isinstance(bias, torch.Tensor) and isinstance(scale, torch.Tensor)):
        raise TypeError("All arguments must be torch.Tensors")
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on XPU")
    if x.dtype != torch.float16 or weight.dtype != torch.float16 \
       or bias.dtype != torch.float16 or scale.dtype != torch.float16:
        raise TypeError("All tensors must be float16")
    if x.ndim != 5 or weight.ndim != 5 or bias.ndim != 1 or scale.ndim != 0:
        raise ValueError("Expected x:5D, weight:5D, bias:1D, scale:0D")

    N, C_in, D, H, W = x.shape
    C_in_w, C_out, KD, KH, KW = weight.shape
    if C_in != C_in_w or KD != 3 or KH != 3 or KW != 3:
        raise ValueError("Unexpected shapes for conv_transpose3d")

    STRIDE = 2
    PAD = 1
    D_out = (D - 1) * STRIDE - 2 * PAD + (KD - 1) + 1
    H_out = (H - 1) * STRIDE - 2 * PAD + (KH - 1) + 1
    W_out = (W - 1) * STRIDE - 2 * PAD + (KW - 1) + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

    xs_n, xs_c, xs_d, xs_h, xs_w = x.stride()
    ws_ci, ws_co, ws_kd, ws_kh, ws_kw = weight.stride()
    ys_n, ys_c, ys_d, ys_h, ys_w = y.stride()

    def grid(meta):
        return (triton.cdiv(D_out * H_out * W_out, meta["BLOCK_SIZE"]), N * C_out)

    _conv_transpose3d_fused_kernel_specialized[grid](
        x, weight, bias, scale, y,
        N, C_out,
        D, H, W,
        D_out, H_out, W_out,
        xs_n, xs_c, xs_d, xs_h, xs_w,
        ws_ci, ws_co, ws_kd, ws_kh, ws_kw,
        ys_n, ys_c, ys_d, ys_h, ys_w,
        C_IN=C_in,
        grf_mode="auto",
    )
    return y


def _max_pool3d_triton(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on XPU")
    if x.ndim != 5:
        raise ValueError("Input must be NCDHW (5D)")
    if x.dtype != torch.float16:
        raise TypeError("Only float16 is supported")

    kd, kh, kw = 2, 2, 2
    sd, sh, sw = 2, 2, 2
    pd, ph, pw = 0, 0, 0
    dd, dh, dw = 1, 1, 1

    N, C, D, H, W = x.shape

    def out_dim(in_size, k, s, p, d):
        eff = (k - 1) * d + 1
        return math.floor((in_size + 2 * p - eff) / s) + 1

    OD = out_dim(D, kd, sd, pd, dd)
    OH = out_dim(H, kh, sh, ph, dh)
    OW = out_dim(W, kw, sw, pw, dw)

    y = torch.empty((N, C, OD, OH, OW), dtype=x.dtype, device=x.device)
    sN, sC, sD, sH, sW = x.stride()
    yN, yC, yD, yH, yW = y.stride()

    def grid(meta):
        return (N * C * OD * OH, triton.cdiv(OW, meta["BLOCK_W"]))

    _max_pool3d_kernel[grid](
        x, y,
        N, C, D, H, W,
        OD, OH, OW,
        sN, sC, sD, sH, sW,
        yN, yC, yD, yH, yW,
        KERNEL_D=kd, KERNEL_H=kh, KERNEL_W=kw,
        STRIDE_D=sd, STRIDE_H=sh, STRIDE_W=sw,
        PAD_D=pd, PAD_H=ph, PAD_W=pw,
        DIL_D=dd, DIL_H=dh, DIL_W=dw,
        grf_mode="auto",
    )
    return y


def _adaptive_avg_pool3d_clamp(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on XPU")
    if x.ndim != 5:
        raise ValueError("Input must be NCDHW (5D)")

    N, C, D, H, W = x.shape
    y = torch.empty((N, C, 1, 1, 1), dtype=x.dtype, device=x.device)
    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oD, oH, oW = y.stride()

    grid = (N * C,)
    _avgpool3d_clamp_ncdhw_1x1x1[grid](
        x, y,
        N, C, D, H, W,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
        grf_mode="auto",
    )
    return y


# -------------------------------------------------------------------
# Top-level composed kernel_function
# -------------------------------------------------------------------
def kernel_function(x, weight, bias, scale):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available")

    x_xpu = x.to("xpu", dtype=torch.float16).contiguous() if (
        x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous()
    ) else x
    weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous() if (
        weight.device.type != "xpu" or weight.dtype != torch.float16 or not weight.is_contiguous()
    ) else weight
    bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous() if (
        bias.device.type != "xpu" or bias.dtype != torch.float16 or not bias.is_contiguous()
    ) else bias
    scale_xpu = scale.to("xpu", dtype=torch.float16).contiguous() if (
        scale.device.type != "xpu" or scale.dtype != torch.float16 or scale.numel() != 1
    ) else scale

    y1 = _conv_transpose3d_mul_scale(x_xpu, weight_xpu, bias_xpu, scale_xpu)
    y2 = _max_pool3d_triton(y1)
    y3 = _adaptive_avg_pool3d_clamp(y2)
    return y3


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=1)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self._params_on_xpu = False

    def _ensure_xpu_params(self):
        if not self._params_on_xpu:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.scale.data = self.scale.data.to("xpu", dtype=torch.float16)
            self._params_on_xpu = True
        else:
            if self.conv_transpose.weight.device.type != "xpu" or self.conv_transpose.weight.dtype != torch.float16 or not self.conv_transpose.weight.is_contiguous():
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
            if self.conv_transpose.bias.device.type != "xpu" or self.conv_transpose.bias.dtype != torch.float16 or not self.conv_transpose.bias.is_contiguous():
                self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float16).contiguous()
            if self.scale.device.type != "xpu" or self.scale.dtype != torch.float16:
                self.scale.data = self.scale.data.to("xpu", dtype=torch.float16)

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to("xpu", dtype=torch.float16).contiguous()
        self._ensure_xpu_params()
        return kernel_function(x, self.conv_transpose.weight, self.conv_transpose.bias, self.scale)