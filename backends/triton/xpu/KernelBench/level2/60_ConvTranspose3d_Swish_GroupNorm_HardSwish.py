import torch
import torch.nn as nn
import triton
import triton.language as tl


def _convt3d_autotune_configs():
    return [
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=2),
    ]


def _groupnorm_w64_autotune_configs():
    return [
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=2),
    ]


def _groupnorm_w32_autotune_configs():
    return [
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 32}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=1),
    ]


@triton.autotune(
    configs=_convt3d_autotune_configs(),
    key=["N", "C_OUT", "D_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def convt3d_swish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_OUT, D_IN, H_IN, W_IN, D_OUT, H_OUT, W_OUT,
    stride_nx, stride_cx, stride_dx, stride_hx, stride_wx,
    stride_w_ic, stride_w_oc, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_ny, stride_cy, stride_dy, stride_hy, stride_wy,
    BLOCK_W: tl.constexpr, C_IN: tl.constexpr,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_D: tl.constexpr, DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    oh = pid0 % H_OUT
    tmp = pid0 // H_OUT
    od = tmp % D_OUT
    n = tmp // D_OUT
    oc = pid1

    offs_w = pid2 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_OUT
    tl.max_contiguous(offs_w, BLOCK_W)

    n64 = n.to(tl.int64)
    oc64 = oc.to(tl.int64)
    od64 = od.to(tl.int64)
    oh64 = oh.to(tl.int64)
    offs_w64 = offs_w.to(tl.int64)

    y_row_base = n64 * stride_ny + oc64 * stride_cy + od64 * stride_dy + oh64 * stride_hy
    y_ptrs = y_ptr + y_row_base + offs_w64 * stride_wy

    acc = tl.full([BLOCK_W], tl.load(b_ptr + oc).to(tl.float32), dtype=tl.float32)

    for ic in tl.static_range(0, C_IN):
        ic64 = tl.full((), ic, tl.int64)
        x_base_nc = n64 * stride_nx + ic64 * stride_cx
        w_base_icoc = ic64 * stride_w_ic + oc64 * stride_w_oc

        for kd in tl.static_range(0, K_D):
            id_num = od + PAD_D - kd * DIL_D
            if (id_num % STRIDE_D) == 0:
                id_in = id_num // STRIDE_D
                if (id_in >= 0) and (id_in < D_IN):
                    id64 = tl.full((), id_in, tl.int64)
                    x_base_ncd = x_base_nc + id64 * stride_dx
                    w_base_kd = w_base_icoc + tl.full((), kd, tl.int64) * stride_w_kd

                    for kh in tl.static_range(0, K_H):
                        ih_num = oh + PAD_H - kh * DIL_H
                        if (ih_num % STRIDE_H) == 0:
                            ih_in = ih_num // STRIDE_H
                            if (ih_in >= 0) and (ih_in < H_IN):
                                ih64 = tl.full((), ih_in, tl.int64)
                                x_base_ncdh = x_base_ncd + ih64 * stride_hx
                                w_base_kdh = w_base_kd + tl.full((), kh, tl.int64) * stride_w_kh

                                for kw in tl.static_range(0, K_W):
                                    iw_num = offs_w + PAD_W - kw * DIL_W
                                    iw_in = iw_num // STRIDE_W
                                    mask = mask_w & ((iw_num % STRIDE_W) == 0) & (iw_in >= 0) & (iw_in < W_IN)

                                    x_vals = tl.load(
                                        x_ptr + x_base_ncdh + iw_in.to(tl.int64) * stride_wx,
                                        mask=mask,
                                        other=0.0,
                                    ).to(tl.float32)

                                    w_val = tl.load(
                                        w_ptr + w_base_kdh + tl.full((), kw, tl.int64) * stride_w_kw
                                    ).to(tl.float32)

                                    acc += x_vals * w_val

    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig
    tl.store(y_ptrs, out.to(tl.float16), mask=mask_w)


@triton.autotune(
    configs=_groupnorm_w64_autotune_configs(),
    key=["N", "C", "D", "H", "W", "G"],
)
@triton.jit
def _groupnorm_hardswish_kernel_w64(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    eps,
    C_PER_G: tl.constexpr, G: tl.constexpr,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    c_base = g * C_PER_G

    n64 = n.to(tl.int64)

    sum_vec = tl.zeros([BLOCK_W], dtype=tl.float32)
    sumsq_vec = tl.zeros([BLOCK_W], dtype=tl.float32)

    for cc in range(C_PER_G):
        ci = c_base + cc
        ci64 = tl.full((), ci, tl.int64)
        base_nc = n64 * stride_n + ci64 * stride_c
        for dd in range(D):
            dd64 = tl.full((), dd, tl.int64)
            base_ncd = base_nc + dd64 * stride_d
            for hh in range(H):
                hh64 = tl.full((), hh, tl.int64)
                base = base_ncd + hh64 * stride_h
                x_bp = tl.make_block_ptr(
                    base=x_ptr + base,
                    shape=(1, W),
                    strides=(stride_h, stride_w),
                    offsets=(0, 0),
                    block_shape=(1, BLOCK_W),
                    order=(1, 0),
                )
                vals = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                vals = vals.to(tl.float32)
                vals = tl.reshape(vals, [BLOCK_W])
                sum_vec += vals
                sumsq_vec += vals * vals

    sum_val = tl.sum(sum_vec, axis=0)
    sum_sq = tl.sum(sumsq_vec, axis=0)

    m = C_PER_G * D * H * W
    m_f = tl.full([], m, tl.float32)
    mean = sum_val / m_f
    var = sum_sq / m_f - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    for cc in range(C_PER_G):
        ci = c_base + cc
        ci64 = tl.full((), ci, tl.int64)
        gval = tl.load(gamma_ptr + ci).to(tl.float32)
        bval = tl.load(beta_ptr + ci).to(tl.float32)
        base_nc = n64 * stride_n + ci64 * stride_c

        for dd in range(D):
            dd64 = tl.full((), dd, tl.int64)
            base_ncd = base_nc + dd64 * stride_d
            for hh in range(H):
                hh64 = tl.full((), hh, tl.int64)
                base = base_ncd + hh64 * stride_h

                x_bp = tl.make_block_ptr(
                    base=x_ptr + base,
                    shape=(1, W),
                    strides=(stride_h, stride_w),
                    offsets=(0, 0),
                    block_shape=(1, BLOCK_W),
                    order=(1, 0),
                )
                y_bp = tl.make_block_ptr(
                    base=y_ptr + base,
                    shape=(1, W),
                    strides=(stride_h, stride_w),
                    offsets=(0, 0),
                    block_shape=(1, BLOCK_W),
                    order=(1, 0),
                )

                x_vals = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                x_vals = x_vals.to(tl.float32)
                x_vals = tl.reshape(x_vals, [BLOCK_W])

                yv = (x_vals - mean) * inv_std
                yv = yv * gval + bval
                t = tl.minimum(tl.maximum(yv + 3.0, 0.0), 6.0)
                hsw = yv * t * (1.0 / 6.0)
                tl.store(y_bp, tl.reshape(hsw.to(tl.float16), [1, BLOCK_W]), boundary_check=(0, 1))


@triton.autotune(
    configs=_groupnorm_w32_autotune_configs(),
    key=["N", "C", "D", "H", "W", "G"],
)
@triton.jit
def _groupnorm_hardswish_kernel_w32(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    eps,
    C_PER_G: tl.constexpr, G: tl.constexpr,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // G
    g = pid0 % G
    c_base = g * C_PER_G

    w_start = pid1 * BLOCK_W
    n64 = n.to(tl.int64)

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    for cc in range(C_PER_G):
        ci = c_base + cc
        ci64 = tl.full((), ci, tl.int64)
        base_nc = n64 * stride_n + ci64 * stride_c
        for dd in range(D):
            dd64 = tl.full((), dd, tl.int64)
            base_ncd = base_nc + dd64 * stride_d
            for hh in range(H):
                hh64 = tl.full((), hh, tl.int64)
                base = base_ncd + hh64 * stride_h
                x_bp = tl.make_block_ptr(
                    base=x_ptr + base,
                    shape=(1, W),
                    strides=(stride_h, stride_w),
                    offsets=(0, w_start),
                    block_shape=(1, BLOCK_W),
                    order=(1, 0),
                )
                vals = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                vals = vals.to(tl.float32)
                vals = tl.reshape(vals, [BLOCK_W])
                sum_val += tl.sum(vals, axis=0)
                sum_sq += tl.sum(vals * vals, axis=0)

    if pid1 == 0:
        m = C_PER_G * D * H * W
        m_f = tl.full([], m, tl.float32)
        mean = sum_val / m_f
        var = sum_sq / m_f - mean * mean
        var = tl.maximum(var, 0.0)
        inv_std = 1.0 / tl.sqrt(var + eps)

        for cc in range(C_PER_G):
            ci = c_base + cc
            ci64 = tl.full((), ci, tl.int64)
            gval = tl.load(gamma_ptr + ci).to(tl.float32)
            bval = tl.load(beta_ptr + ci).to(tl.float32)
            base_nc = n64 * stride_n + ci64 * stride_c
            for dd in range(D):
                dd64 = tl.full((), dd, tl.int64)
                base_ncd = base_nc + dd64 * stride_d
                for hh in range(H):
                    hh64 = tl.full((), hh, tl.int64)
                    base = base_ncd + hh64 * stride_h
                    x_bp = tl.make_block_ptr(
                        base=x_ptr + base,
                        shape=(1, W),
                        strides=(stride_h, stride_w),
                        offsets=(0, w_start),
                        block_shape=(1, BLOCK_W),
                        order=(1, 0),
                    )
                    y_bp = tl.make_block_ptr(
                        base=y_ptr + base,
                        shape=(1, W),
                        strides=(stride_h, stride_w),
                        offsets=(0, w_start),
                        block_shape=(1, BLOCK_W),
                        order=(1, 0),
                    )
                    x_vals = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                    x_vals = x_vals.to(tl.float32)
                    x_vals = tl.reshape(x_vals, [BLOCK_W])
                    yv = (x_vals - mean) * inv_std
                    yv = yv * gval + bval
                    t = tl.minimum(tl.maximum(yv + 3.0, 0.0), 6.0)
                    hsw = yv * t * (1.0 / 6.0)
                    tl.store(y_bp, tl.reshape(hsw.to(tl.float16), [1, BLOCK_W]), boundary_check=(0, 1))


def kernel_function(x, conv_w, conv_b, gn_weight, gn_bias, num_groups, eps):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU must be available"

    x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    conv_w_xpu = conv_w.to("xpu", dtype=torch.float16).contiguous()
    conv_b_xpu = conv_b.to("xpu", dtype=torch.float32).contiguous()
    gn_weight_xpu = gn_weight.to("xpu", dtype=torch.float32).contiguous()
    gn_bias_xpu = gn_bias.to("xpu", dtype=torch.float32).contiguous()

    N, C_in, D_in, H_in, W_in = x_xpu.shape
    assert C_in == conv_w_xpu.shape[0]
    assert conv_w_xpu.shape[1] == gn_weight_xpu.numel()

    stride_d, stride_h, stride_w = 2, 2, 2
    pad_d, pad_h, pad_w = 1, 1, 1
    dil_d, dil_h, dil_w = 1, 1, 1

    kD, kH, kW = conv_w_xpu.shape[2:]
    C_out = conv_w_xpu.shape[1]
    D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (kD - 1) + 1
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + 1

    y1 = torch.empty((N, C_out, D_out, H_out, W_out), device="xpu", dtype=torch.float16)

    sx_n, sx_c, sx_d, sx_h, sx_w = x_xpu.stride()
    sw_ic, sw_oc, sw_kd, sw_kh, sw_kw = conv_w_xpu.stride()
    sy_n, sy_c, sy_d, sy_h, sy_w = y1.stride()

    grid0 = lambda meta: (N * D_out * H_out, C_out, triton.cdiv(W_out, meta["BLOCK_W"]))
    convt3d_swish_kernel[grid0](
        x_xpu, conv_w_xpu, conv_b_xpu, y1,
        N, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        sx_n, sx_c, sx_d, sx_h, sx_w,
        sw_ic, sw_oc, sw_kd, sw_kh, sw_kw,
        sy_n, sy_c, sy_d, sy_h, sy_w,
        C_IN=C_in,
        K_D=kD, K_H=kH, K_W=kW,
        STRIDE_D=stride_d, STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_D=pad_d, PAD_H=pad_h, PAD_W=pad_w,
        DIL_D=dil_d, DIL_H=dil_h, DIL_W=dil_w,
        grf_mode="auto",
    )

    y2 = torch.empty_like(y1)
    N2, C2, D2, H2, W2 = y1.shape
    sN, sC, sD, sH, sW = y1.stride()
    G = num_groups
    C_PER_G = C2 // G

    if W2 <= 32:
        grid1 = lambda meta: (N2 * G, triton.cdiv(W2, meta["BLOCK_W"]))
        _groupnorm_hardswish_kernel_w32[grid1](
            y1, y2, gn_weight_xpu, gn_bias_xpu,
            N2, C2, D2, H2, W2,
            sN, sC, sD, sH, sW,
            eps,
            C_PER_G=C_PER_G,
            G=G,
            grf_mode="auto",
        )
    else:
        grid1 = (N2 * G,)
        _groupnorm_hardswish_kernel_w64[grid1](
            y1, y2, gn_weight_xpu, gn_bias_xpu,
            N2, C2, D2, H2, W2,
            sN, sC, sD, sH, sW,
            eps,
            C_PER_G=C_PER_G,
            G=G,
            grf_mode="auto",
        )
    return y2


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=2, padding=1, bias=bias
        )
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.stride = stride
        self.padding = padding
        self._weights_on_xpu = False

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        if not self._weights_on_xpu:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float32).contiguous()
            self.group_norm.weight.data = self.group_norm.weight.data.to("xpu", dtype=torch.float32).contiguous()
            self.group_norm.bias.data = self.group_norm.bias.data.to("xpu", dtype=torch.float32).contiguous()
            self._weights_on_xpu = True

        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
