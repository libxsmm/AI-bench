# ruff: noqa: E731
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1


def get_inputs():
    return [
        torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float16)
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        divisor,
        pool_size,
        bias_shape,
        sum_dim,
    ]


def _conv3d_autotune_configs():
    return [
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 128}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 128}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 128}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 64}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=32, num_stages=3),
        triton.Config({"BLOCK_H": 256, "BLOCK_W": 256}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 256, "BLOCK_W": 256}, num_warps=32, num_stages=3),
    ]


def _pool_autotune_configs():
    return [
        triton.Config({"BLOCK_OW": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OW": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_OW": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OW": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OW": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OW": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_OW": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_OW": 128}, num_warps=16, num_stages=3),
    ]


def _bias_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=3),
    ]


def _sum_autotune_configs():
    return [
        triton.Config({"BLOCK_N": 32, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_C": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_C": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_C": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_C": 16}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_C": 32}, num_warps=16, num_stages=3),
    ]


@triton.autotune(
    configs=_conv3d_autotune_configs(),
    key=["N", "C_OUT", "D_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def _conv3d_bias_div_wtile_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C_IN,
    D_IN,
    H_IN,
    W_IN,
    C_OUT,
    D_OUT,
    H_OUT,
    W_OUT,
    stride_xn,
    stride_xc,
    stride_xd,
    stride_xh,
    stride_xw,
    stride_wo,
    stride_wi,
    stride_wkd,
    stride_wkh,
    stride_wkw,
    stride_yn,
    stride_yc,
    stride_yd,
    stride_yh,
    stride_yw,
    alpha,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_ncoz = tl.program_id(axis=2)

    co = pid_ncoz % C_OUT
    tmp = pid_ncoz // C_OUT
    zo = tmp % D_OUT
    n = tmp // D_OUT

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_w = offs_w < W_OUT
    mask_h = offs_h < H_OUT
    out_mask = mask_h[:, None] & mask_w[None, :]

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for ci in range(C_IN):
        base_x_n_ci_zo = n * stride_xn + ci * stride_xc + zo * stride_xd
        base_w_co_ci = co * stride_wo + ci * stride_wi
        for kd in range(KD):
            base_x_d = base_x_n_ci_zo + kd * stride_xd
            base_w_kd = base_w_co_ci + kd * stride_wkd
            for kh in range(KH):
                base_x_h = base_x_d + (offs_h[:, None] + kh) * stride_xh
                base_w_kh = base_w_kd + kh * stride_wkh
                for kw in range(KW):
                    w_val = tl.load(w_ptr + base_w_kh + kw * stride_wkw)
                    x_ptrs = x_ptr + base_x_h + (offs_w[None, :] + kw) * stride_xw
                    in_bounds = (
                        out_mask
                        & ((offs_h[:, None] + kh) < H_IN)
                        & ((offs_w[None, :] + kw) < W_IN)
                    )
                    x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)
                    acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc = (acc + b_val) * alpha

    y_bp = tl.make_block_ptr(
        base=y_ptr + n * stride_yn + co * stride_yc + zo * stride_yd,
        shape=(H_OUT, W_OUT),
        strides=(stride_yh, stride_yw),
        offsets=(pid_h * BLOCK_H, pid_w * BLOCK_W),
        block_shape=(BLOCK_H, BLOCK_W),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


def _conv3d_bias_div(x, w, b, divisor=2.0):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU not available.")
    assert x.device.type == "xpu" and w.device.type == "xpu" and b.device.type == "xpu"
    assert (
        x.dtype == torch.float16
        and w.dtype == torch.float16
        and b.dtype == torch.float16
    )

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, Cw_in, kD, kH, kW = w.shape
    assert C_in == Cw_in and b.shape[0] == C_out

    D_out = D_in - (kD - 1)
    H_out = H_in - (kH - 1)
    W_out = W_in - (kW - 1)

    y = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

    sxn, sxc, sxd, sxh, sxw = x.stride()
    swo, swi, swkd, swkh, swkw = w.stride()
    syn, syc, syd, syh, syw = y.stride()

    alpha = float(1.0 / divisor)

    def grid(meta):
        return (
            triton.cdiv(W_out, meta["BLOCK_W"]),
            triton.cdiv(H_out, meta["BLOCK_H"]),
            N * C_out * D_out,
        )

    _conv3d_bias_div_wtile_kernel[grid](
        x,
        w,
        b,
        y,
        N,
        C_in,
        D_in,
        H_in,
        W_in,
        C_out,
        D_out,
        H_out,
        W_out,
        sxn,
        sxc,
        sxd,
        sxh,
        sxw,
        swo,
        swi,
        swkd,
        swkh,
        swkw,
        syn,
        syc,
        syd,
        syh,
        syw,
        alpha,
        KD=kD,
        KH=kH,
        KW=kW,
        grf_mode="auto",
    )
    return y


@triton.autotune(
    configs=_pool_autotune_configs(),
    key=["N", "C", "D_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def _fused_maxpool3d_adaptive_avgpool3d_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    sN,
    sC,
    sD,
    sH,
    sW,
    ysN,
    ysC,
    ysD,
    ysH,
    ysW,
    D_OUT,
    H_OUT,
    W_OUT,
    scale,
    BLOCK_OW: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    base_nc = n * sN + c * sC
    acc_sum = tl.zeros((), dtype=tl.float32)
    ow_ramp = tl.arange(0, BLOCK_OW)

    for od in tl.range(0, D_OUT):
        d0 = od * 2
        for oh in tl.range(0, H_OUT):
            h0 = oh * 2
            base_dh = base_nc + d0 * sD + h0 * sH
            for ow_start in tl.range(0, W_OUT, BLOCK_OW):
                ow = ow_start + ow_ramp
                ow_mask = ow < W_OUT
                w0 = ow * 2
                ptr000 = x_ptr + base_dh + w0 * sW

                x000 = tl.load(ptr000, mask=ow_mask, other=0.0).to(tl.float32)
                x001 = tl.load(ptr000 + sW, mask=ow_mask, other=0.0).to(tl.float32)
                x010 = tl.load(ptr000 + sH, mask=ow_mask, other=0.0).to(tl.float32)
                x011 = tl.load(ptr000 + sH + sW, mask=ow_mask, other=0.0).to(tl.float32)
                x100 = tl.load(ptr000 + sD, mask=ow_mask, other=0.0).to(tl.float32)
                x101 = tl.load(ptr000 + sD + sW, mask=ow_mask, other=0.0).to(tl.float32)
                x110 = tl.load(ptr000 + sD + sH, mask=ow_mask, other=0.0).to(tl.float32)
                x111 = tl.load(ptr000 + sD + sH + sW, mask=ow_mask, other=0.0).to(
                    tl.float32
                )

                m0 = tl.maximum(x000, x001)
                m1 = tl.maximum(x010, x011)
                m2 = tl.maximum(x100, x101)
                m3 = tl.maximum(x110, x111)
                m4 = tl.maximum(m0, m1)
                m5 = tl.maximum(m2, m3)
                max8 = tl.maximum(m4, m5)
                acc_sum += tl.sum(max8 * ow_mask.to(tl.float32), axis=0)

    avg_f32 = acc_sum * scale
    out_offset = n * ysN + c * ysC
    tl.store(y_ptr + out_offset, avg_f32.to(y_ptr.dtype.element_ty))


def _fused_maxpool3d_adaptive_avgpool3d(x):
    assert x.device.type == "xpu"
    N, C, D, H, W = x.shape
    D_OUT, H_OUT, W_OUT = D // 2, H // 2, W // 2

    y = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

    sN, sC, sD, sH, sW = x.stride()
    ysN, ysC, ysD, ysH, ysW = y.stride()
    total = D_OUT * H_OUT * W_OUT
    scale = float(1.0 / total)
    grid = (N * C,)

    _fused_maxpool3d_adaptive_avgpool3d_kernel[grid](
        x,
        y,
        N,
        C,
        D,
        H,
        W,
        sN,
        sC,
        sD,
        sH,
        sW,
        ysN,
        ysC,
        ysD,
        ysH,
        ysW,
        D_OUT,
        H_OUT,
        W_OUT,
        scale,
        grf_mode="auto",
    )
    return y


@triton.autotune(
    configs=_bias_autotune_configs(),
    key=["n_elements", "C"],
)
@triton.jit
def _add_bias_broadcast_kernel(
    x_ptr,
    b_ptr,
    y_ptr,
    n_elements,
    C,
    stride_xn,
    stride_xc,
    stride_b0,
    stride_yn,
    stride_yc,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c_idx = offsets % C
    n_idx = offsets // C

    x_idx = n_idx * stride_xn + c_idx * stride_xc
    y_idx = n_idx * stride_yn + c_idx * stride_yc
    b_idx = c_idx * stride_b0

    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0)
    b_val = tl.load(b_ptr + b_idx, mask=mask, other=0)
    y_f32 = x_val.to(tl.float32) + b_val.to(tl.float32)
    tl.store(y_ptr + y_idx, y_f32.to(y_ptr.dtype.element_ty), mask=mask)


def _add_bias_broadcast(x0, x1):
    assert x0.device.type == x1.device.type == "xpu"
    assert x0.dtype == x1.dtype

    N, C = x0.shape[0], x0.shape[1]
    y = torch.empty_like(x0)

    n_elements = N * C
    stride_xn, stride_xc = x0.stride(0), x0.stride(1)
    stride_b0 = x1.stride(0)
    stride_yn, stride_yc = y.stride(0), y.stride(1)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _add_bias_broadcast_kernel[grid](
        x0,
        x1,
        y,
        n_elements,
        C,
        stride_xn,
        stride_xc,
        stride_b0,
        stride_yn,
        stride_yc,
        grf_mode="auto",
    )
    return y


@triton.autotune(
    configs=_sum_autotune_configs(),
    key=["N", "C"],
)
@triton.jit
def _sum_dim1_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    stride_n,
    stride_c,
    out_stride_n,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n0 = pid * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, C),
        strides=(stride_n, stride_c),
        offsets=(n0, 0),
        block_shape=(BLOCK_N, BLOCK_C),
        order=(1, 0),
    )
    vals = tl.load(x_bp, boundary_check=(0, 1))
    acc = tl.sum(vals.to(tl.float32), axis=1)

    out_ptrs = y_ptr + offs_n * out_stride_n
    tl.store(out_ptrs, acc.to(y_ptr.dtype.element_ty), mask=offs_n < N)


def _sum_dim1(x):
    assert x.device.type == "xpu" and x.dtype == torch.float16
    N, C = x.shape[0], x.shape[1]
    y = torch.empty((N,) + x.shape[2:], dtype=x.dtype, device=x.device)

    stride_n, stride_c = x.stride(0), x.stride(1)
    out_stride_n = y.stride(0)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    _sum_dim1_kernel[grid](
        x,
        y,
        N,
        C,
        stride_n,
        stride_c,
        out_stride_n,
        grf_mode="auto",
    )
    return y


def kernel_function(x, conv_w, conv_b, bias):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU not available.")

    x_xpu = (
        x
        if (x.device.type == "xpu" and x.dtype == torch.float16)
        else x.to("xpu", dtype=torch.float16)
    )
    conv_w_xpu = (
        conv_w
        if (conv_w.device.type == "xpu" and conv_w.dtype == torch.float16)
        else conv_w.to("xpu", dtype=torch.float16)
    )
    conv_b_xpu = (
        conv_b
        if (conv_b.device.type == "xpu" and conv_b.dtype == torch.float16)
        else conv_b.to("xpu", dtype=torch.float16)
    )
    bias_xpu = (
        bias
        if (bias.device.type == "xpu" and bias.dtype == torch.float16)
        else bias.to("xpu", dtype=torch.float16)
    )

    y1 = _conv3d_bias_div(x_xpu, conv_w_xpu, conv_b_xpu, divisor=2.0)
    y2 = _fused_maxpool3d_adaptive_avgpool3d(y1)
    y3 = _add_bias_broadcast(y2, bias_xpu)
    y4 = _sum_dim1(y3)
    torch.xpu.synchronize()
    return y4


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        divisor,
        pool_size,
        bias_shape,
        sum_dim,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.zeros(bias_shape))
        self.divisor = divisor
        self.pool_size = pool_size
        self.sum_dim = sum_dim

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if (
            self.conv.weight.device.type != "xpu"
            or self.conv.weight.dtype != torch.float16
        ):
            self.conv.weight.data = self.conv.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        if self.conv.bias is not None and (
            self.conv.bias.device.type != "xpu" or self.conv.bias.dtype != torch.float16
        ):
            self.conv.bias.data = self.conv.bias.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        if self.bias.device.type != "xpu" or self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()

        return kernel_function(x, self.conv.weight, self.conv.bias, self.bias)


def run_test():
    init_args = get_init_inputs()
    model = Model(*init_args).eval()

    (x,) = get_inputs()
    x_ref = x.to("xpu", dtype=torch.float16)

    with torch.no_grad():
        conv = nn.Conv3d(in_channels, out_channels, kernel_size).to(
            "xpu", dtype=torch.float16
        )
        conv.weight.copy_(model.conv.weight.to("xpu", dtype=torch.float16))
        conv.bias.copy_(model.conv.bias.to("xpu", dtype=torch.float16))
        b = model.bias.to("xpu", dtype=torch.float16)

        ref = conv(x_ref)
        ref = ref / divisor
        ref = torch.nn.functional.max_pool3d(ref, pool_size)
        ref = torch.nn.functional.adaptive_avg_pool3d(ref, (1, 1, 1))
        ref = ref + b
        ref = torch.sum(ref, dim=sum_dim)

    out = model(x)

    if torch.allclose(out, ref, rtol=1e-3, atol=1e-3):
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        print("Max abs diff:", (out - ref).abs().max().item())
        sys.exit(1)
