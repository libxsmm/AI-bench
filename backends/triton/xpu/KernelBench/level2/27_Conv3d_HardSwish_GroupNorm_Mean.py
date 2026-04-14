# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 16, "BLOCK_CO": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 32, "BLOCK_CO": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 32, "BLOCK_CO": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 64, "BLOCK_CO": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 64, "BLOCK_CO": 16}, num_warps=8, num_stages=2),
    ],
    key=["W_OUT", "C_OUT"],
)
@triton.jit
def _conv3d_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, D, H, W,
    C_OUT, D_OUT, H_OUT, W_OUT,
    SXN, SXC, SXD, SXH, SXW,
    SWO, SWC, SWD, SWH, SWW,
    SYN, SYC, SYD, SYH, SYW,
    Kd: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_CO: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_dh = tl.program_id(axis=1)
    pid_nc = tl.program_id(axis=2)

    h_out = pid_dh % H_OUT
    d_out = pid_dh // H_OUT

    co_groups = tl.cdiv(C_OUT, BLOCK_CO)
    n_idx = pid_nc // co_groups
    co_group = pid_nc % co_groups

    n_idx_i64 = n_idx.to(tl.int64)
    d_out_i64 = d_out.to(tl.int64)
    h_out_i64 = h_out.to(tl.int64)

    w_start = pid_w * BLOCK_W
    co_start = co_group * BLOCK_CO

    offs_w = w_start + tl.arange(0, BLOCK_W)
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    mask_w = offs_w < W_OUT
    mask_co = offs_co < C_OUT

    acc = tl.zeros((BLOCK_CO, BLOCK_W), dtype=tl.float32)

    base_x = x_ptr + n_idx_i64 * SXN + d_out_i64 * SXD + h_out_i64 * SXH

    for ci in tl.static_range(0, 3):
        ci_i64 = tl.full((), ci, tl.int64)
        x_ci = base_x + ci_i64 * SXC
        w_ci = w_ptr + ci_i64 * SWC

        for kd in tl.static_range(0, Kd):
            kd_i64 = tl.full((), kd, tl.int64)
            x_kd = x_ci + kd_i64 * SXD
            w_kd = w_ci + kd_i64 * SWD

            for kh in tl.static_range(0, Kh):
                kh_i64 = tl.full((), kh, tl.int64)
                x_row = x_kd + kh_i64 * SXH
                w_row = w_kd + kh_i64 * SWH

                for kw in tl.static_range(0, Kw):
                    kw_i64 = tl.full((), kw, tl.int64)
                    x_vals = tl.load(
                        x_row + (offs_w.to(tl.int64) + kw_i64) * SXW,
                        mask=mask_w,
                        other=0.0,
                    ).to(tl.float32)
                    w_vals = tl.load(
                        w_row + offs_co.to(tl.int64) * SWO + kw_i64 * SWW,
                        mask=mask_co,
                        other=0.0,
                    ).to(tl.float32)
                    acc += w_vals[:, None] * x_vals[None, :]

    bv = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc += bv[:, None]

    t = acc + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    acc = acc * (t * (1.0 / 6.0))

    y_base = y_ptr + n_idx_i64 * SYN + d_out_i64 * SYD + h_out_i64 * SYH
    y_ptrs = y_base + offs_co[:, None].to(tl.int64) * SYC + offs_w[None, :].to(tl.int64) * SYW
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_co[:, None] & mask_w[None, :])


@triton.jit
def _groupnorm_ncdhw_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    N, C, D, H, W,
    NUM_GROUPS, CPG,
    eps, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    n_i64 = n.to(tl.int64)
    g_i64 = g.to(tl.int64)

    spatial = stride_c
    group_len = CPG * spatial
    c0 = g * CPG
    c0_i64 = g_i64 * CPG

    x_base = x_ptr + n_i64 * stride_n + c0_i64 * stride_c
    y_base = y_ptr + n_i64 * stride_n + c0_i64 * stride_c

    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    for off in tl.range(0, group_len, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_len
        x_vals = tl.load(x_base + offs.to(tl.int64), mask=mask, other=0.0)
        x_f32 = x_vals.to(tl.float32)
        sum_x += tl.sum(x_f32, axis=0)
        sum_x2 += tl.sum(x_f32 * x_f32, axis=0)

    group_len_f32 = tl.full((), group_len, dtype=tl.float32)
    mean = sum_x / group_len_f32
    var = sum_x2 / group_len_f32 - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    for off in tl.range(0, group_len, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_len
        x_vals = tl.load(x_base + offs.to(tl.int64), mask=mask, other=0.0)
        x_f32 = x_vals.to(tl.float32)
        y_f32 = (x_f32 - mean) * inv_std

        ch_in_group = offs // spatial
        ch_idx = c0 + ch_in_group
        gamma = tl.load(weight_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(bias_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)

        y_f32 = y_f32 * gamma + beta
        tl.store(y_base + offs.to(tl.int64), y_f32.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_mean_dhw_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    n_i64 = n.to(tl.int64)
    c_i64 = c.to(tl.int64)

    base_ptr = x_ptr + n_i64 * stride_n + c_i64 * stride_c
    acc = tl.zeros((), dtype=tl.float32)

    offs_w = tl.arange(0, BLOCK_W)
    for d in tl.range(0, D):
        d_i64 = d.to(tl.int64)
        for h in tl.range(0, H):
            h_i64 = h.to(tl.int64)
            row_base = base_ptr + d_i64 * stride_d + h_i64 * stride_h
            for w0 in tl.range(0, W, BLOCK_W):
                offs = w0 + offs_w
                mask = offs < W
                vals = tl.load(row_base + offs.to(tl.int64) * stride_w, mask=mask, other=0.0)
                acc += tl.sum(vals.to(tl.float32), axis=0)

    denom = tl.full((), D * H * W, dtype=tl.float32)
    mean_val = acc / denom
    tl.store(out_ptr + n_i64 * out_stride_n + c_i64 * out_stride_c, mean_val.to(out_ptr.dtype.element_ty))


def kernel_function(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU not available.")

    x_xpu = x if x.device.type == "xpu" and x.dtype == torch.float16 else x.to("xpu", dtype=torch.float16)
    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()

    conv_w_xpu = conv_w if conv_w.device.type == "xpu" and conv_w.dtype == torch.float16 else conv_w.to("xpu", dtype=torch.float16)
    if not conv_w_xpu.is_contiguous():
        conv_w_xpu = conv_w_xpu.contiguous()

    conv_b_xpu = conv_b if conv_b.device.type == "xpu" and conv_b.dtype == torch.float16 else conv_b.to("xpu", dtype=torch.float16)
    if not conv_b_xpu.is_contiguous():
        conv_b_xpu = conv_b_xpu.contiguous()

    gn_weight_xpu = gn_weight if gn_weight.device.type == "xpu" and gn_weight.dtype == torch.float16 else gn_weight.to("xpu", dtype=torch.float16)
    if not gn_weight_xpu.is_contiguous():
        gn_weight_xpu = gn_weight_xpu.contiguous()

    gn_bias_xpu = gn_bias if gn_bias.device.type == "xpu" and gn_bias.dtype == torch.float16 else gn_bias.to("xpu", dtype=torch.float16)
    if not gn_bias_xpu.is_contiguous():
        gn_bias_xpu = gn_bias_xpu.contiguous()

    N, C_in, D, H, W = x_xpu.shape
    C_out, Cw_in, Kd, Kh, Kw = conv_w_xpu.shape
    assert Cw_in == C_in
    assert conv_b_xpu.shape == (C_out,)

    D_out = D - Kd + 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1

    y1 = torch.empty((N, C_out, D_out, H_out, W_out), device=x_xpu.device, dtype=torch.float16)

    SXN, SXC, SXD, SXH, SXW = x_xpu.stride()
    SWO, SWC, SWD, SWH, SWW = conv_w_xpu.stride()
    SYN, SYC, SYD, SYH, SYW = y1.stride()

    grid_conv = (
        triton.cdiv(W_out, 64),
        D_out * H_out,
        N * triton.cdiv(C_out, 16),
    )
    _conv3d_hardswish_kernel[grid_conv](
        x_xpu, conv_w_xpu, conv_b_xpu, y1,
        N, C_in, D, H, W,
        C_out, D_out, H_out, W_out,
        SXN, SXC, SXD, SXH, SXW,
        SWO, SWC, SWD, SWH, SWW,
        SYN, SYC, SYD, SYH, SYW,
        Kd=Kd, Kh=Kh, Kw=Kw,
        grf_mode="auto",
    )

    N2, C2, D2, H2, W2 = y1.shape
    num_groups = 4
    assert C2 == gn_weight_xpu.shape[0] == gn_bias_xpu.shape[0]
    assert C2 % num_groups == 0
    CPG = C2 // num_groups

    y2 = torch.empty_like(y1)
    sn, sc, sd, sh, sw = y1.stride()
    _groupnorm_ncdhw_kernel[(N2 * num_groups,)](
        y1, y2, gn_weight_xpu, gn_bias_xpu,
        sn, sc, sd, sh, sw,
        N2, C2, D2, H2, W2,
        num_groups, CPG,
        float(1e-5),
        BLOCK_SIZE=1024,
        num_warps=8,
        num_stages=2,
    )

    y3 = torch.empty((N2, C2), device=x_xpu.device, dtype=torch.float16)
    sN, sC, sD, sH, sW = y2.stride()
    oN, oC = y3.stride()
    _reduce_mean_dhw_kernel[(N2 * C2,)](
        y2, y3,
        N2, C2, D2, H2, W2,
        sN, sC, sD, sH, sW,
        oN, oC,
        BLOCK_W=32,
        num_warps=8,
        num_stages=2,
    )

    return y3


batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self._xpu_prepared = False

    def _prepare_for_xpu(self):
        if self.conv.weight.device.type != "xpu" or self.conv.weight.dtype != torch.float16:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
        else:
            self.conv.weight.data = self.conv.weight.data.contiguous()

        if self.conv.bias is not None:
            if self.conv.bias.device.type != "xpu" or self.conv.bias.dtype != torch.float16:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            else:
                self.conv.bias.data = self.conv.bias.data.contiguous()

        if self.group_norm.weight.device.type != "xpu" or self.group_norm.weight.dtype != torch.float16:
            self.group_norm.weight.data = self.group_norm.weight.data.to("xpu", dtype=torch.float16).contiguous()
        else:
            self.group_norm.weight.data = self.group_norm.weight.data.contiguous()

        if self.group_norm.bias.device.type != "xpu" or self.group_norm.bias.dtype != torch.float16:
            self.group_norm.bias.data = self.group_norm.bias.data.to("xpu", dtype=torch.float16).contiguous()
        else:
            self.group_norm.bias.data = self.group_norm.bias.data.contiguous()

        self._xpu_prepared = True

    def forward(self, x):
        if not self._xpu_prepared:
            self._prepare_for_xpu()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()
        return kernel_function(
            x,
            self.conv.weight,
            self.conv.bias,
            self.group_norm.weight,
            self.group_norm.bias,
        )