# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _convtranspose3d_autotune_configs():
    configs = []
    # Focused but broadened XPU-oriented sweep.
    # Keep stage count modest to avoid autotune overhead and register collapse.
    base = [
        (32, 64, [(4, 1), (8, 1)]),
        (32, 128, [(4, 1), (8, 1)]),
        (64, 64, [(4, 1), (8, 1), (16, 1)]),
        (64, 128, [(8, 1), (16, 1)]),
        (64, 256, [(8, 1), (16, 1)]),
        (128, 64, [(8, 1), (16, 1)]),
        (128, 128, [(8, 1), (16, 1), (32, 1)]),
        (128, 256, [(16, 1), (32, 1)]),
        (256, 128, [(16, 1), (32, 1)]),
        (256, 256, [(32, 1)]),  # required large-tile 32-warp XPU config
    ]
    for block_co, block_w, warp_stage_choices in base:
        for num_warps, num_stages in warp_stage_choices:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_CO": block_co,
                        "BLOCK_W": block_w,
                    },
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


def _mul_autotune_configs():
    configs = []
    # Pointwise kernel: larger vectors and 16/32 warp options for XPU throughput.
    for block_size, warp_stage_choices in [
        (128, [(4, 1), (8, 1)]),
        (256, [(4, 1), (8, 1)]),
        (512, [(4, 1), (8, 1), (16, 1)]),
        (1024, [(8, 1), (16, 1)]),
        (2048, [(16, 1), (32, 1)]),
    ]:
        for num_warps, num_stages in warp_stage_choices:
            configs.append(
                triton.Config(
                    {"BLOCK_SIZE": block_size},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


def _pool_autotune_configs():
    configs = []
    # Row-wise pool kernel: include small fallback and larger XPU-friendly blocks.
    for block_ow, warp_stage_choices in [
        (64, [(4, 1), (8, 1)]),
        (128, [(4, 1), (8, 1)]),
        (256, [(8, 1), (16, 1)]),
        (512, [(16, 1), (32, 1)]),
    ]:
        for num_warps, num_stages in warp_stage_choices:
            configs.append(
                triton.Config(
                    {"BLOCK_OW": block_ow},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


# ---------------------------------------------------------------------
# Triton kernel for ConvTranspose3d + Bias + LeakyReLU
# ---------------------------------------------------------------------
@triton.autotune(
    configs=_convtranspose3d_autotune_configs(),
    key=["C_IN", "C_OUT", "D_OUT", "H_OUT", "W_OUT", "K_D", "K_H", "K_W"],
)
@triton.jit
def _convtranspose3d_leakyrelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C_IN,
    C_OUT,
    D_IN,
    H_IN,
    W_IN,
    D_OUT,
    H_OUT,
    W_OUT,
    STRIDE_D,
    STRIDE_H,
    STRIDE_W,
    PAD_D,
    PAD_H,
    PAD_W,
    x_stride_n,
    x_stride_c,
    x_stride_d,
    x_stride_h,
    x_stride_w,
    w_stride_ci,
    w_stride_co,
    w_stride_kd,
    w_stride_kh,
    w_stride_kw,
    y_stride_n,
    y_stride_c,
    y_stride_d,
    y_stride_h,
    y_stride_w,
    NEG_SLOPE: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_W: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_ndh = tl.program_id(axis=1)
    pid_co = tl.program_id(axis=2)

    oh = pid_ndh % H_OUT
    tmp = pid_ndh // H_OUT
    od = tmp % D_OUT
    n = tmp // D_OUT

    n64 = n.to(tl.int64)
    od64 = od.to(tl.int64)
    oh64 = oh.to(tl.int64)

    w_start = pid_w * BLOCK_W
    co_start = pid_co * BLOCK_CO
    offs_w = w_start + tl.arange(0, BLOCK_W)
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    mask_w = offs_w < W_OUT
    mask_co = offs_co < C_OUT

    offs_w64 = offs_w.to(tl.int64)
    offs_co64 = offs_co.to(tl.int64)

    acc = tl.zeros((BLOCK_CO, BLOCK_W), dtype=tl.float32)

    x_batch_base = x_ptr + n64 * x_stride_n
    y_batch_base = y_ptr + n64 * y_stride_n
    y_row_base = y_batch_base + od64 * y_stride_d + oh64 * y_stride_h

    for ci in range(0, C_IN):
        ci64 = tl.full((), ci, tl.int64)
        x_ci_base = x_batch_base + ci64 * x_stride_c
        w_ci_base = w_ptr + ci64 * w_stride_ci

        for kd in range(0, K_D):
            rd = od + PAD_D - kd
            cond_d = (rd % STRIDE_D) == 0
            id_in = rd // STRIDE_D
            valid_d = cond_d & (id_in >= 0) & (id_in < D_IN)
            if valid_d:
                id64 = tl.full((), id_in, tl.int64)
                x_d_base = x_ci_base + id64 * x_stride_d
                w_kd_base = w_ci_base + tl.full((), kd, tl.int64) * w_stride_kd

                for kh in range(0, K_H):
                    rh = oh + PAD_H - kh
                    cond_h = (rh % STRIDE_H) == 0
                    ih_in = rh // STRIDE_H
                    valid_h = cond_h & (ih_in >= 0) & (ih_in < H_IN)
                    if valid_h:
                        ih64 = tl.full((), ih_in, tl.int64)
                        x_h_base = x_d_base + ih64 * x_stride_h
                        w_kh_base = w_kd_base + tl.full((), kh, tl.int64) * w_stride_kh

                        for kw in range(0, K_W):
                            rw = offs_w + PAD_W - kw
                            cond_w = (rw % STRIDE_W) == 0
                            iw_in = rw // STRIDE_W
                            mask_vec = mask_w & cond_w & (iw_in >= 0) & (iw_in < W_IN)

                            x_ptrs = x_h_base + iw_in.to(tl.int64) * x_stride_w
                            x_vals = tl.load(x_ptrs, mask=mask_vec, other=0.0).to(
                                tl.float32
                            )

                            w_base = w_kh_base + tl.full((), kw, tl.int64) * w_stride_kw
                            w_ptrs = w_base + offs_co64 * w_stride_co
                            w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0).to(
                                tl.float32
                            )

                            acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + offs_co64, mask=mask_co, other=0.0).to(tl.float32)
    acc = acc + b_vals[:, None]
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    y_ptrs = (
        y_row_base + offs_co64[:, None] * y_stride_c + offs_w64[None, :] * y_stride_w
    )
    store_mask = mask_co[:, None] & mask_w[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------
# Triton kernel for Mul + LeakyReLU
# ---------------------------------------------------------------------
@triton.autotune(
    configs=_mul_autotune_configs(),
    key=["n_elements", "C", "W"],
)
@triton.jit
def _mul_leakyrelu_5d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    C,
    D,
    H,
    W,
    x_stride_n,
    x_stride_c,
    x_stride_d,
    x_stride_h,
    x_stride_w,
    w_stride_c,
    neg_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    w_idx = offsets % W
    tmp = offsets // W
    h_idx = tmp % H
    tmp = tmp // H
    d_idx = tmp % D
    tmp = tmp // D
    c_idx = tmp % C
    n_idx = tmp // C

    x_offs = (
        n_idx.to(tl.int64) * x_stride_n
        + c_idx.to(tl.int64) * x_stride_c
        + d_idx.to(tl.int64) * x_stride_d
        + h_idx.to(tl.int64) * x_stride_h
        + w_idx.to(tl.int64) * x_stride_w
    )
    w_offs = c_idx.to(tl.int64) * w_stride_c

    x_val = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
    w_val = tl.load(w_ptr + w_offs, mask=mask, other=0.0)

    y_f32 = x_val.to(tl.float32) * w_val.to(tl.float32)
    y_f32 = tl.where(y_f32 >= 0, y_f32, y_f32 * neg_slope)

    tl.store(out_ptr + x_offs, y_f32.to(x_val.dtype), mask=mask)


# ---------------------------------------------------------------------
# Triton kernel for MaxPool3d k=2, s=2, p=0
# ---------------------------------------------------------------------
@triton.autotune(
    configs=_pool_autotune_configs(),
    key=["OW", "OH", "OD", "C"],
)
@triton.jit
def _maxpool3d_k2s2_p0_rowwise(
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
    strideN,
    strideC,
    strideD,
    strideH,
    strideW,
    out_strideN,
    out_strideC,
    out_strideD,
    out_strideH,
    out_strideW,
    BLOCK_OW: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    oh = pid_row % OH
    tmp = pid_row // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    ow_start = pid_col * BLOCK_OW
    ow_offsets = ow_start + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offsets < OW

    id0 = od * 2
    ih0 = oh * 2
    iw0 = ow_offsets * 2

    base_nc = n.to(tl.int64) * strideN + c.to(tl.int64) * strideC

    d0_in = id0 < D
    d1_in = (id0 + 1) < D
    h0_in = ih0 < H
    h1_in = (ih0 + 1) < H

    neg_inf = -float("inf")
    iw064 = iw0.to(tl.int64)
    id064 = tl.full((), id0, tl.int64)
    ih064 = tl.full((), ih0, tl.int64)

    ptr000 = x_ptr + (
        base_nc + id064 * strideD + ih064 * strideH + (iw064 + 0) * strideW
    )
    mask000 = ow_mask & tl.full(ow_mask.shape, d0_in & h0_in, tl.int1) & ((iw0 + 0) < W)
    maxv = tl.load(ptr000, mask=mask000, other=neg_inf)

    ptr001 = x_ptr + (
        base_nc + id064 * strideD + ih064 * strideH + (iw064 + 1) * strideW
    )
    mask001 = ow_mask & tl.full(ow_mask.shape, d0_in & h0_in, tl.int1) & ((iw0 + 1) < W)
    maxv = tl.maximum(maxv, tl.load(ptr001, mask=mask001, other=neg_inf))

    ptr010 = x_ptr + (
        base_nc + id064 * strideD + (ih064 + 1) * strideH + (iw064 + 0) * strideW
    )
    mask010 = ow_mask & tl.full(ow_mask.shape, d0_in & h1_in, tl.int1) & ((iw0 + 0) < W)
    maxv = tl.maximum(maxv, tl.load(ptr010, mask=mask010, other=neg_inf))

    ptr011 = x_ptr + (
        base_nc + id064 * strideD + (ih064 + 1) * strideH + (iw064 + 1) * strideW
    )
    mask011 = ow_mask & tl.full(ow_mask.shape, d0_in & h1_in, tl.int1) & ((iw0 + 1) < W)
    maxv = tl.maximum(maxv, tl.load(ptr011, mask=mask011, other=neg_inf))

    ptr100 = x_ptr + (
        base_nc + (id064 + 1) * strideD + ih064 * strideH + (iw064 + 0) * strideW
    )
    mask100 = ow_mask & tl.full(ow_mask.shape, d1_in & h0_in, tl.int1) & ((iw0 + 0) < W)
    maxv = tl.maximum(maxv, tl.load(ptr100, mask=mask100, other=neg_inf))

    ptr101 = x_ptr + (
        base_nc + (id064 + 1) * strideD + ih064 * strideH + (iw064 + 1) * strideW
    )
    mask101 = ow_mask & tl.full(ow_mask.shape, d1_in & h0_in, tl.int1) & ((iw0 + 1) < W)
    maxv = tl.maximum(maxv, tl.load(ptr101, mask=mask101, other=neg_inf))

    ptr110 = x_ptr + (
        base_nc + (id064 + 1) * strideD + (ih064 + 1) * strideH + (iw064 + 0) * strideW
    )
    mask110 = ow_mask & tl.full(ow_mask.shape, d1_in & h1_in, tl.int1) & ((iw0 + 0) < W)
    maxv = tl.maximum(maxv, tl.load(ptr110, mask=mask110, other=neg_inf))

    ptr111 = x_ptr + (
        base_nc + (id064 + 1) * strideD + (ih064 + 1) * strideH + (iw064 + 1) * strideW
    )
    mask111 = ow_mask & tl.full(ow_mask.shape, d1_in & h1_in, tl.int1) & ((iw0 + 1) < W)
    maxv = tl.maximum(maxv, tl.load(ptr111, mask=mask111, other=neg_inf))

    y_base = y_ptr + (
        n.to(tl.int64) * out_strideN
        + c.to(tl.int64) * out_strideC
        + od.to(tl.int64) * out_strideD
        + oh.to(tl.int64) * out_strideH
    )
    y_bp = tl.make_block_ptr(
        base=y_base,
        shape=(1, OW),
        strides=(out_strideH, out_strideW),
        offsets=(0, ow_start),
        block_shape=(1, BLOCK_OW),
        order=(1, 0),
    )
    tl.store(y_bp, maxv[None, :], boundary_check=(0, 1))


def _all_ones_multiplier(multiplier: torch.Tensor) -> bool:
    if multiplier.numel() == 0:
        return False
    return bool(torch.all(multiplier == 1).item())


# ---------------------------------------------------------------------
# Composite kernel_function
# ---------------------------------------------------------------------
def kernel_function(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, multiplier: torch.Tensor
) -> torch.Tensor:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU device is not available")

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be torch.Tensor")
    if not isinstance(w, torch.Tensor):
        raise TypeError("w must be torch.Tensor")
    if not isinstance(b, torch.Tensor):
        raise TypeError("b must be torch.Tensor")
    if not isinstance(multiplier, torch.Tensor):
        raise TypeError("multiplier must be torch.Tensor")

    x_xpu = x if x.device.type == "xpu" else x.to("xpu")
    w_xpu = w if w.device.type == "xpu" else w.to("xpu")
    b_xpu = b if b.device.type == "xpu" else b.to("xpu")
    multiplier_xpu = (
        multiplier if multiplier.device.type == "xpu" else multiplier.to("xpu")
    )

    N, C_in, D_in, H_in, W_in = x_xpu.shape
    Ci_w, Co_w, Kd, Kh, Kw = w_xpu.shape
    assert Ci_w == C_in, "Weight C_in mismatch"
    C_out = Co_w
    assert b_xpu.numel() == C_out, "Bias length mismatch"

    Sd, Sh, Sw = 2, 2, 2
    Pd, Ph, Pw = 1, 1, 1
    Opd, Oph, Opw = 1, 1, 1

    D_out = (D_in - 1) * Sd - 2 * Pd + (Kd - 1) + Opd + 1
    H_out = (H_in - 1) * Sh - 2 * Ph + (Kh - 1) + Oph + 1
    W_out = (W_in - 1) * Sw - 2 * Pw + (Kw - 1) + Opw + 1

    y1 = torch.empty(
        (N, C_out, D_out, H_out, W_out), device=x_xpu.device, dtype=x_xpu.dtype
    )
    x_strides = x_xpu.stride()
    w_strides = w_xpu.stride()
    y1_strides = y1.stride()

    grid1 = lambda META: (
        triton.cdiv(W_out, META["BLOCK_W"]),
        N * D_out * H_out,
        triton.cdiv(C_out, META["BLOCK_CO"]),
    )
    _convtranspose3d_leakyrelu_kernel[grid1](
        x_xpu,
        w_xpu,
        b_xpu,
        y1,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        Sd,
        Sh,
        Sw,
        Pd,
        Ph,
        Pw,
        x_strides[0],
        x_strides[1],
        x_strides[2],
        x_strides[3],
        x_strides[4],
        w_strides[0],
        w_strides[1],
        w_strides[2],
        w_strides[3],
        w_strides[4],
        y1_strides[0],
        y1_strides[1],
        y1_strides[2],
        y1_strides[3],
        y1_strides[4],
        NEG_SLOPE=0.2,
        K_D=Kd,
        K_H=Kh,
        K_W=Kw,
        grf_mode="auto",
    )

    if _all_ones_multiplier(multiplier_xpu):
        x3 = y1
    else:
        N2, C2, D2, H2, W2 = y1.shape
        assert multiplier_xpu.shape == (C2, 1, 1, 1), "Multiplier shape mismatch"
        out2 = torch.empty_like(y1)
        x2_strides = y1.stride()
        w_stride_c = multiplier_xpu.stride(0)
        n_elements = y1.numel()

        grid2 = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _mul_leakyrelu_5d_kernel[grid2](
            y1,
            multiplier_xpu,
            out2,
            N2,
            C2,
            D2,
            H2,
            W2,
            x2_strides[0],
            x2_strides[1],
            x2_strides[2],
            x2_strides[3],
            x2_strides[4],
            w_stride_c,
            0.2,
            n_elements,
            grf_mode="auto",
        )
        x3 = out2

    N3, C3, D3, H3, W3 = x3.shape
    OD, OH, OW = D3 // 2, H3 // 2, W3 // 2
    y3 = torch.empty((N3, C3, OD, OH, OW), device=x3.device, dtype=x3.dtype)
    sN, sC, sD, sH, sW = x3.stride()
    oN, oC, oD, oH, oW = y3.stride()
    rows = N3 * C3 * OD * OH

    grid3 = lambda META: (rows, triton.cdiv(OW, META["BLOCK_OW"]))
    _maxpool3d_k2s2_p0_rowwise[grid3](
        x3,
        y3,
        N3,
        C3,
        D3,
        H3,
        W3,
        OD,
        OH,
        OW,
        sN,
        sC,
        sD,
        sH,
        sW,
        oN,
        oC,
        oD,
        oH,
        oW,
        grf_mode="auto",
    )
    return y3


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        multiplier_shape,
    ]


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        multiplier_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=1,
            output_padding=output_padding,
        )
        self.multiplier = nn.Parameter(torch.ones(multiplier_shape))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if (
            self.conv_transpose.weight.device.type != "xpu"
            or self.conv_transpose.weight.dtype != torch.float16
        ):
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        if (
            self.conv_transpose.bias.device.type != "xpu"
            or self.conv_transpose.bias.dtype != torch.float16
        ):
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        if (
            self.multiplier.device.type != "xpu"
            or self.multiplier.dtype != torch.float16
        ):
            self.multiplier.data = self.multiplier.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()

        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.multiplier,
        )
