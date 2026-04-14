# ruff: noqa: E731
# KernelBench-compatible wrapper — keep all original Triton kernels available.
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Subgraph 1: ConvTranspose3d + bias + broadcast add
# -------------------------------------------------------------------------
@triton.jit
def _conv_transpose3d_add_kernel(
    x_ptr, w_ptr, b_ptr, add_ptr, y_ptr,
    N, Cin, Cout, D, H, W, outD, outH, outW,
    stride_xN, stride_xC, stride_xD, stride_xH, stride_xW,
    stride_wCIN, stride_wCOUT, stride_wKD, stride_wKH, stride_wKW,
    stride_yN, stride_yC, stride_yD, stride_yH, stride_yW,
    stride_addN, stride_addC, stride_addD, stride_addH, stride_addW,
    SD, SH, SW, PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_nzy = tl.program_id(2)

    ox_offsets = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ox_mask = ox_offsets < outW
    co_mask = co_offsets < Cout

    OH = outH
    OD = outD
    nzy_total = OD * OH
    n = pid_nzy // nzy_total
    rem = pid_nzy % nzy_total
    oz = rem // OH
    oy = rem % OH

    acc = tl.zeros((BLOCK_CO, BLOCK_X), dtype=tl.float32)

    for ci in tl.range(0, Cin):
        for kd in range(KD):
            tzd = oz + PD - kd
            z_div = (tzd % SD) == 0
            iz = tzd // SD
            valid_z = z_div & (iz >= 0) & (iz < D)
            for kh in range(KH):
                tyd = oy + PH - kh
                y_div = (tyd % SH) == 0
                iy = tyd // SH
                valid_y = y_div & (iy >= 0) & (iy < H)
                zy_valid = valid_z & valid_y
                for kw in range(KW):
                    tx = ox_offsets + PW - kw
                    x_div = (tx % SW) == 0
                    ix = tx // SW
                    x_in_range = (ix >= 0) & (ix < W)
                    mask = ox_mask & x_div & x_in_range & zy_valid

                    base_x = n * stride_xN + ci * stride_xC + iz * stride_xD + iy * stride_xH
                    x_ptrs = x_ptr + base_x + ix * stride_xW
                    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

                    w_ptrs = (
                        w_ptr
                        + ci * stride_wCIN
                        + co_offsets * stride_wCOUT
                        + kd * stride_wKD
                        + kh * stride_wKH
                        + kw * stride_wKW
                    )
                    w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                    acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + co_offsets * stride_addC, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals[:, None] + add_vals[:, None]

    y_base = n * stride_yN + oz * stride_yD + oy * stride_yH
    y_ptrs = y_ptr + y_base + co_offsets[:, None] * stride_yC + ox_offsets[None, :] * stride_yW
    mask2 = co_mask[:, None] & ox_mask[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask2)


def conv_transpose3d_add(
    x, w, b, sum_weight,
    stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), groups=1
):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available"
    assert x.is_xpu and w.is_xpu and b.is_xpu and sum_weight.is_xpu, "tensors must be on xpu"
    assert groups == 1, "only groups=1 supported"
    N, Cin, D, H, W = x.shape
    Cout = w.shape[1]
    kD, kH, kW = w.shape[2:]
    SD, SH, SW = stride
    PD, PH, PW = padding
    OPD, OPH, OPW = output_padding
    outD = (D - 1) * SD - 2 * PD + kD + OPD
    outH = (H - 1) * SH - 2 * PH + kH + OPH
    outW = (W - 1) * SW - 2 * PW + kW + OPW

    y = torch.empty((N, Cout, outD, outH, outW), device=x.device, dtype=x.dtype)
    sx = x.stride()
    sw = w.stride()
    sy = y.stride()
    sa = sum_weight.stride()
    grid = (
        triton.cdiv(outW, 64),
        triton.cdiv(Cout, 32),
        N * outD * outH,
    )
    _conv_transpose3d_add_kernel[grid](
        x, w, b, sum_weight, y,
        N, Cin, Cout, D, H, W, outD, outH, outW,
        sx[0], sx[1], sx[2], sx[3], sx[4],
        sw[0], sw[1], sw[2], sw[3], sw[4],
        sy[0], sy[1], sy[2], sy[3], sy[4],
        sa[0], sa[1], sa[2], sa[3], sa[4],
        SD, SH, SW, PD, PH, PW,
        KD=kD, KH=kH, KW=kW,
        BLOCK_CO=32, BLOCK_X=64,
        num_warps=8, num_stages=2
    )
    return y


# -------------------------------------------------------------------------
# Subgraph 2: LayerNorm over last dim
# -------------------------------------------------------------------------
@triton.jit
def _layernorm_lastdim_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    M, N, eps, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / N
    xc = x_f32 - mean
    var = tl.sum(xc * xc, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    yv = xc * inv_std
    if HAS_WEIGHT:
        g = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        yv = yv * g
    if HAS_BIAS:
        bb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        yv = yv + bb
    y_cast = yv.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row_start + offs, y_cast, mask=mask)


def layernorm(x, weight, bias, eps=1e-5):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available"
    assert x.is_xpu and weight.is_xpu and bias.is_xpu, "tensors must be on xpu"
    N_last = x.shape[-1]
    x_contig = x.contiguous() if not x.is_contiguous() else x
    w = weight.contiguous() if not weight.is_contiguous() else weight
    b = bias.contiguous() if not bias.is_contiguous() else bias
    M = x_contig.numel() // N_last
    y = torch.empty_like(x_contig)
    BLOCK = N_last
    grid = (M,)
    _layernorm_lastdim_kernel[grid](
        x_contig, y, w, b,
        M, N_last, eps, HAS_WEIGHT=True, HAS_BIAS=True, BLOCK_SIZE=BLOCK,
        num_warps=4, num_stages=1
    )
    return y.view_as(x)


# -------------------------------------------------------------------------
# Subgraph 3: AvgPool3d k=2 s=2 + GELU
# -------------------------------------------------------------------------
@triton.jit
def _gelu_via_erf_approx(x):
    inv_sqrt2 = 0.7071067811865476
    z = x * inv_sqrt2
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    az = tl.abs(z)
    t = 1.0 / (1.0 + p * az)
    poly = a5
    poly = poly * t + a4
    poly = poly * t + a3
    poly = poly * t + a2
    poly = poly * t + a1
    poly = poly * t
    e = tl.exp(-az * az)
    erf_approx = 1.0 - poly * e
    sign = tl.where(z >= 0, 1.0, -1.0)
    erf_val = sign * erf_approx
    return 0.5 * x * (1.0 + erf_val)


@triton.jit
def _avgpool3d_k2s2_gelu_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W, DO, HO, WO,
    sN, sC, sD, sH, sW,
    oN, oC, oD, oH, oW,
    BLOCK_W: tl.constexpr
):
    pid_nc = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_wt = tl.program_id(2)
    n = pid_nc // C
    c = pid_nc % C
    do = pid_dh // HO
    ho = pid_dh % HO
    w_start = pid_wt * BLOCK_W
    w_off = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_off < WO

    d0 = 2 * do
    h0 = 2 * ho
    w0 = 2 * w_off
    base0 = n * sN + c * sC + d0 * sD + h0 * sH + w0 * sW

    p000 = base0
    p001 = p000 + sW
    p010 = p000 + sH
    p011 = p010 + sW
    p100 = p000 + sD
    p101 = p100 + sW
    p110 = p100 + sH
    p111 = p110 + sW

    x000 = tl.load(x_ptr + p000, mask=mask_w, other=0.0).to(tl.float32)
    x001 = tl.load(x_ptr + p001, mask=mask_w, other=0.0).to(tl.float32)
    x010 = tl.load(x_ptr + p010, mask=mask_w, other=0.0).to(tl.float32)
    x011 = tl.load(x_ptr + p011, mask=mask_w, other=0.0).to(tl.float32)
    x100 = tl.load(x_ptr + p100, mask=mask_w, other=0.0).to(tl.float32)
    x101 = tl.load(x_ptr + p101, mask=mask_w, other=0.0).to(tl.float32)
    x110 = tl.load(x_ptr + p110, mask=mask_w, other=0.0).to(tl.float32)
    x111 = tl.load(x_ptr + p111, mask=mask_w, other=0.0).to(tl.float32)

    acc = x000 + x001 + x010 + x011 + x100 + x101 + x110 + x111
    avg = acc * (1.0 / 8.0)
    out = _gelu_via_erf_approx(avg)

    out_ptr = y_ptr + n * oN + c * oC + do * oD + ho * oH + w_off * oW
    tl.store(out_ptr, out.to(y_ptr.dtype.element_ty), mask=mask_w)


def avgpool3d_k2s2_gelu(x):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available"
    assert x.is_xpu, "x must be on xpu"
    N, C, D, H, W = x.shape
    DO, HO, WO = D // 2, H // 2, W // 2
    y = torch.empty((N, C, DO, HO, WO), device=x.device, dtype=x.dtype)
    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oD, oH, oW = y.stride()
    BLOCK_W = 128
    grid = (N * C, DO * HO, triton.cdiv(WO, BLOCK_W))
    _avgpool3d_k2s2_gelu_kernel[grid](
        x, y,
        N, C, D, H, W, DO, HO, WO,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
        BLOCK_W=BLOCK_W,
        num_warps=8, num_stages=2
    )
    return y


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _to_xpu_contig(t, dtype=None):
    if dtype is None:
        dtype = t.dtype
    if t.device.type != "xpu" or t.dtype != dtype:
        t = t.to("xpu", dtype=dtype)
    return t.contiguous()


# -------------------------------------------------------------------------
# Combined kernel_function
# -------------------------------------------------------------------------
def kernel_function(x, conv_w, conv_b, sum_weight, ln_weight, ln_bias):
    """
    End-to-end forward matching the current implementation's semantics:
      conv_transpose3d -> LayerNorm(over last dim as coded) -> AvgPool3d -> GELU

    DTYPE_FIX optimization:
      - only runtime input x is normalized here
      - parameters are expected to already be on XPU with stable cached dtypes
      - avoid repeated per-call parameter conversion/check overhead
    """
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU not available"

    x_xpu = _to_xpu_contig(x, torch.float16)

    y1 = F.conv_transpose3d(
        x_xpu, conv_w, conv_b,
        stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)
    )
    y2 = layernorm(y1, ln_weight, ln_bias, eps=1e-5)
    y3 = avgpool3d_k2s2_gelu(y2)
    return y3


# -------------------------------------------------------------------------
# Original model and helpers for testing
# -------------------------------------------------------------------------
batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = sum_weight
        out_c = out_channels
        self.layer_norm = nn.LayerNorm(out_c)
        self.norm_shape = norm_shape
        self.pool_kernel_size = pool_kernel_size

        self._cached_sum_weight_xpu = None
        self._cached_sum_meta = None
        self._xpu_params_ready = False

    def _ensure_xpu_state(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous()

        if not self._xpu_params_ready:
            if self.conv_transpose.weight.device.type != "xpu" or self.conv_transpose.weight.dtype != torch.float16:
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.conv_transpose.weight.is_contiguous():
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous()

            if self.conv_transpose.bias is not None:
                desired_bias_dtype = self.conv_transpose.bias.dtype if self.conv_transpose.bias.dtype == torch.float16 else torch.float32
                if self.conv_transpose.bias.device.type != "xpu" or self.conv_transpose.bias.dtype != desired_bias_dtype:
                    self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=desired_bias_dtype).contiguous()
                elif not self.conv_transpose.bias.is_contiguous():
                    self.conv_transpose.bias.data = self.conv_transpose.bias.data.contiguous()

            if self.layer_norm.weight.device.type != "xpu":
                self.layer_norm.weight.data = self.layer_norm.weight.data.to("xpu").contiguous()
            elif not self.layer_norm.weight.is_contiguous():
                self.layer_norm.weight.data = self.layer_norm.weight.data.contiguous()

            if self.layer_norm.bias.device.type != "xpu":
                self.layer_norm.bias.data = self.layer_norm.bias.data.to("xpu").contiguous()
            elif not self.layer_norm.bias.is_contiguous():
                self.layer_norm.bias.data = self.layer_norm.bias.data.contiguous()

            self._xpu_params_ready = True

        if isinstance(self.sum_weight, (int, float)):
            meta = (x.device.type, x.dtype, float(self.sum_weight))
            if self._cached_sum_weight_xpu is None or self._cached_sum_meta != meta:
                self._cached_sum_weight_xpu = torch.tensor(float(self.sum_weight), device="xpu", dtype=x.dtype)
                self._cached_sum_meta = meta
            sum_weight = self._cached_sum_weight_xpu
        else:
            if self.sum_weight.device.type != "xpu" or self.sum_weight.dtype != x.dtype:
                self.sum_weight = self.sum_weight.to("xpu", dtype=x.dtype).contiguous()
            elif not self.sum_weight.is_contiguous():
                self.sum_weight = self.sum_weight.contiguous()
            sum_weight = self.sum_weight

        return x, sum_weight

    def forward(self, x):
        x_xpu, sum_weight_xpu = self._ensure_xpu_state(x)
        return kernel_function(
            x_xpu,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            sum_weight_xpu,
            self.layer_norm.weight,
            self.layer_norm.bias,
        )