import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]


def _conv_autotune_configs():
    return [
        # Small tiles
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_warps=8, num_stages=2),
        # Medium / balanced
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=16, num_stages=3),
        # Large XPU-oriented tiles
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 256}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 256, "BLOCK_W": 128}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_H": 256, "BLOCK_W": 256}, num_warps=32, num_stages=2),
    ]


def _group_norm_autotune_configs():
    return [
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_W": 512}, num_warps=16, num_stages=2),
    ]


@triton.autotune(
    configs=_conv_autotune_configs(),
    key=["Cin", "Cout", "Hout", "Wout"],
)
@triton.jit
def conv_transpose2d_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N, Cin, Hin, Win, Cout, Kh, Kw, Hout, Wout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // Cout
    oc = pid_nc % Cout

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    offs_h = oh_start + tl.arange(0, BLOCK_H)
    offs_w = ow_start + tl.arange(0, BLOCK_W)
    mask_h = offs_h < Hout
    mask_w = offs_w < Wout
    mask_hw = mask_h[:, None] & mask_w[None, :]

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    y_base_noc = n * stride_yn + oc * stride_yc
    x_base_n = n * stride_xn
    w_base_oc = oc * stride_wc

    for ic in range(0, Cin):
        x_base = x_base_n + ic * stride_xc
        w_base = ic * stride_wn + w_base_oc

        ih0 = offs_h[:, None] - 0
        iw0 = offs_w[None, :] - 0
        m0 = mask_hw & (ih0 >= 0) & (ih0 < Hin) & (iw0 >= 0) & (iw0 < Win)
        x0 = tl.load(x_ptr + x_base + ih0 * stride_xh + iw0 * stride_xw, mask=m0, other=0.0)
        w0 = tl.load(w_ptr + w_base + 0 * stride_wh + 0 * stride_ww)
        acc += x0 * w0

        ih1 = offs_h[:, None] - 0
        iw1 = offs_w[None, :] - 1
        m1 = mask_hw & (ih1 >= 0) & (ih1 < Hin) & (iw1 >= 0) & (iw1 < Win)
        x1 = tl.load(x_ptr + x_base + ih1 * stride_xh + iw1 * stride_xw, mask=m1, other=0.0)
        w1 = tl.load(w_ptr + w_base + 0 * stride_wh + 1 * stride_ww)
        acc += x1 * w1

        ih2 = offs_h[:, None] - 0
        iw2 = offs_w[None, :] - 2
        m2 = mask_hw & (ih2 >= 0) & (ih2 < Hin) & (iw2 >= 0) & (iw2 < Win)
        x2 = tl.load(x_ptr + x_base + ih2 * stride_xh + iw2 * stride_xw, mask=m2, other=0.0)
        w2 = tl.load(w_ptr + w_base + 0 * stride_wh + 2 * stride_ww)
        acc += x2 * w2

        ih3 = offs_h[:, None] - 1
        iw3 = offs_w[None, :] - 0
        m3 = mask_hw & (ih3 >= 0) & (ih3 < Hin) & (iw3 >= 0) & (iw3 < Win)
        x3 = tl.load(x_ptr + x_base + ih3 * stride_xh + iw3 * stride_xw, mask=m3, other=0.0)
        w3 = tl.load(w_ptr + w_base + 1 * stride_wh + 0 * stride_ww)
        acc += x3 * w3

        ih4 = offs_h[:, None] - 1
        iw4 = offs_w[None, :] - 1
        m4 = mask_hw & (ih4 >= 0) & (ih4 < Hin) & (iw4 >= 0) & (iw4 < Win)
        x4 = tl.load(x_ptr + x_base + ih4 * stride_xh + iw4 * stride_xw, mask=m4, other=0.0)
        w4 = tl.load(w_ptr + w_base + 1 * stride_wh + 1 * stride_ww)
        acc += x4 * w4

        ih5 = offs_h[:, None] - 1
        iw5 = offs_w[None, :] - 2
        m5 = mask_hw & (ih5 >= 0) & (ih5 < Hin) & (iw5 >= 0) & (iw5 < Win)
        x5 = tl.load(x_ptr + x_base + ih5 * stride_xh + iw5 * stride_xw, mask=m5, other=0.0)
        w5 = tl.load(w_ptr + w_base + 1 * stride_wh + 2 * stride_ww)
        acc += x5 * w5

        ih6 = offs_h[:, None] - 2
        iw6 = offs_w[None, :] - 0
        m6 = mask_hw & (ih6 >= 0) & (ih6 < Hin) & (iw6 >= 0) & (iw6 < Win)
        x6 = tl.load(x_ptr + x_base + ih6 * stride_xh + iw6 * stride_xw, mask=m6, other=0.0)
        w6 = tl.load(w_ptr + w_base + 2 * stride_wh + 0 * stride_ww)
        acc += x6 * w6

        ih7 = offs_h[:, None] - 2
        iw7 = offs_w[None, :] - 1
        m7 = mask_hw & (ih7 >= 0) & (ih7 < Hin) & (iw7 >= 0) & (iw7 < Win)
        x7 = tl.load(x_ptr + x_base + ih7 * stride_xh + iw7 * stride_xw, mask=m7, other=0.0)
        w7 = tl.load(w_ptr + w_base + 2 * stride_wh + 1 * stride_ww)
        acc += x7 * w7

        ih8 = offs_h[:, None] - 2
        iw8 = offs_w[None, :] - 2
        m8 = mask_hw & (ih8 >= 0) & (ih8 < Hin) & (iw8 >= 0) & (iw8 < Win)
        x8 = tl.load(x_ptr + x_base + ih8 * stride_xh + iw8 * stride_xw, mask=m8, other=0.0)
        w8 = tl.load(w_ptr + w_base + 2 * stride_wh + 2 * stride_ww)
        acc += x8 * w8

    b_val = tl.load(b_ptr + oc).to(tl.float32)
    acc += b_val

    inv_sqrt2 = 0.7071067811865476
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    y_bp = tl.make_block_ptr(
        base=y_ptr + y_base_noc,
        shape=(Hout, Wout),
        strides=(stride_yh, stride_yw),
        offsets=(oh_start, ow_start),
        block_shape=(BLOCK_H, BLOCK_W),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


@triton.autotune(
    configs=_group_norm_autotune_configs(),
    key=["C", "H", "W", "num_groups"],
)
@triton.jit
def group_norm_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    beta_ptr,
    N, C, H, W, num_groups, eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    C_per_g = C // num_groups
    c0 = g * C_per_g
    c1 = c0 + C_per_g

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c in range(c0, c1):
        base_off = n * stride_xn + c * stride_xc
        for h in range(0, H):
            row_base = x_ptr + base_off + h * stride_xh
            x_row_bp = tl.make_block_ptr(
                base=row_base,
                shape=(W,),
                strides=(stride_xw,),
                offsets=(0,),
                block_shape=(BLOCK_W,),
                order=(0,),
            )
            for _ in range(0, W, BLOCK_W):
                x_vals = tl.load(x_row_bp, boundary_check=(0,), padding_option="zero").to(tl.float32)
                sum_val += tl.sum(x_vals, axis=0)
                sum_sq += tl.sum(x_vals * x_vals, axis=0)
                x_row_bp = tl.advance(x_row_bp, (BLOCK_W,))

    elem_cnt = C_per_g * H * W
    inv_elem = 1.0 / elem_cnt
    mean = sum_val * inv_elem
    var = sum_sq * inv_elem - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    for c in range(c0, c1):
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        base_off_x = n * stride_xn + c * stride_xc
        base_off_y = n * stride_yn + c * stride_yc
        for h in range(0, H):
            x_row_bp = tl.make_block_ptr(
                base=x_ptr + base_off_x + h * stride_xh,
                shape=(W,),
                strides=(stride_xw,),
                offsets=(0,),
                block_shape=(BLOCK_W,),
                order=(0,),
            )
            y_row_bp = tl.make_block_ptr(
                base=y_ptr + base_off_y + h * stride_yh,
                shape=(W,),
                strides=(stride_yw,),
                offsets=(0,),
                block_shape=(BLOCK_W,),
                order=(0,),
            )
            for _ in range(0, W, BLOCK_W):
                x_vals = tl.load(x_row_bp, boundary_check=(0,), padding_option="zero").to(tl.float32)
                y_vals = (x_vals - mean) * rstd
                y_vals = y_vals * gamma + beta
                tl.store(y_row_bp, y_vals.to(tl.float16), boundary_check=(0,))
                x_row_bp = tl.advance(x_row_bp, (BLOCK_W,))
                y_row_bp = tl.advance(y_row_bp, (BLOCK_W,))


def kernel_function(x, conv_w, conv_b, gn_weight, gn_bias, num_groups):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU unavailable"

    if x.device.type != "xpu" or x.dtype != torch.float16:
        x = x.to("xpu", dtype=torch.float16)
    if conv_w.device.type != "xpu" or conv_w.dtype != torch.float16:
        conv_w = conv_w.to("xpu", dtype=torch.float16)
    if conv_b.device.type != "xpu" or conv_b.dtype != torch.float16:
        conv_b = conv_b.to("xpu", dtype=torch.float16)
    if gn_weight.device.type != "xpu" or gn_weight.dtype != torch.float16:
        gn_weight = gn_weight.to("xpu", dtype=torch.float16)
    if gn_bias.device.type != "xpu" or gn_bias.dtype != torch.float16:
        gn_bias = gn_bias.to("xpu", dtype=torch.float16)

    x = x.contiguous()
    conv_w = conv_w.contiguous()
    conv_b = conv_b.contiguous()
    gn_weight = gn_weight.contiguous()
    gn_bias = gn_bias.contiguous()

    N, Cin, Hin, Win = x.shape
    Cout = conv_w.shape[1]
    Kh, Kw = conv_w.shape[2], conv_w.shape[3]
    Hout = Hin + Kh - 1
    Wout = Win + Kw - 1

    y_act = torch.empty((N, Cout, Hout, Wout), dtype=x.dtype, device=x.device)
    y_out = torch.empty((N, Cout, Hout, Wout), dtype=x.dtype, device=x.device)

    sxn, sxc, sxh, sxw = x.stride()
    swn, swc, swh, sww = conv_w.stride()
    syn, syc, syh, syw = y_act.stride()
    sgn_xn, sgn_xc, sgn_xh, sgn_xw = y_act.stride()
    sgn_yn, sgn_yc, sgn_yh, sgn_yw = y_out.stride()

    grid_conv = lambda meta: (
        triton.cdiv(Wout, meta["BLOCK_W"]),
        triton.cdiv(Hout, meta["BLOCK_H"]),
        N * Cout,
    )
    conv_transpose2d_gelu_kernel[grid_conv](
        x, conv_w, conv_b, y_act,
        N, Cin, Hin, Win, Cout, Kh, Kw, Hout, Wout,
        sxn, sxc, sxh, sxw,
        swn, swc, swh, sww,
        syn, syc, syh, syw,
        grf_mode="auto",
    )

    eps = 1e-5
    grid_gn = (N * num_groups,)
    group_norm_kernel[grid_gn](
        y_act, y_out, gn_weight, gn_bias,
        N, Cout, Hout, Wout, num_groups, eps,
        sgn_xn, sgn_xc, sgn_xh, sgn_xw,
        sgn_yn, sgn_yc, sgn_yh, sgn_yw,
        grf_mode="auto",
    )

    return y_out


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)

        conv_w = self.conv_transpose.weight
        conv_b = self.conv_transpose.bias
        gn_weight = self.group_norm.weight
        gn_bias = self.group_norm.bias

        if conv_w.device.type != "xpu" or conv_w.dtype != torch.float16:
            conv_w = conv_w.to("xpu", dtype=torch.float16)
        if conv_b.device.type != "xpu" or conv_b.dtype != torch.float16:
            conv_b = conv_b.to("xpu", dtype=torch.float16)
        if gn_weight.device.type != "xpu" or gn_weight.dtype != torch.float16:
            gn_weight = gn_weight.to("xpu", dtype=torch.float16)
        if gn_bias.device.type != "xpu" or gn_bias.dtype != torch.float16:
            gn_bias = gn_bias.to("xpu", dtype=torch.float16)

        conv_w = conv_w.contiguous()
        conv_b = conv_b.contiguous()
        gn_weight = gn_weight.contiguous()
        gn_bias = gn_bias.contiguous()

        return kernel_function(
            x,
            conv_w,
            conv_b,
            gn_weight,
            gn_bias,
            self.group_norm.num_groups,
        )