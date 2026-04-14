# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_transpose3d_autotune_configs():
    configs = []

    # Small / medium tiles
    for block_co, block_w, block_ci in [
        (32, 32, 16),
        (32, 64, 16),
        (64, 32, 16),
        (64, 64, 16),
        (32, 128, 16),
        (64, 128, 16),
        (128, 32, 16),
        (128, 64, 16),
    ]:
        for num_warps in (4, 8, 16):
            for num_stages in (1, 2, 3):
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_CO": block_co,
                            "BLOCK_W": block_w,
                            "BLOCK_CI": block_ci,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    # Larger reduction/channel tiles
    for block_co, block_w, block_ci in [
        (64, 64, 32),
        (64, 128, 32),
        (128, 64, 32),
        (128, 128, 32),
    ]:
        for num_warps in (8, 16, 32):
            for num_stages in (1, 2, 3):
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_CO": block_co,
                            "BLOCK_W": block_w,
                            "BLOCK_CI": block_ci,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    # Required large-tile XPU coverage: include 256x256 with 32 warps
    for block_ci in (16, 32):
        for num_stages in (1, 2):
            configs.append(
                triton.Config(
                    {
                        "BLOCK_CO": 256,
                        "BLOCK_W": 256,
                        "BLOCK_CI": block_ci,
                    },
                    num_warps=32,
                    num_stages=num_stages,
                )
            )

    return configs


def _ln_gelu_autotune_configs():
    configs = []

    for rows_per_prog in (8, 16, 32, 64, 128):
        for num_warps in (4, 8, 16):
            for num_stages in (1, 2, 3):
                configs.append(
                    triton.Config(
                        {
                            "ROWS_PER_PROG": rows_per_prog,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    for rows_per_prog in (64, 128, 256):
        for num_stages in (1, 2):
            configs.append(
                triton.Config(
                    {
                        "ROWS_PER_PROG": rows_per_prog,
                    },
                    num_warps=32,
                    num_stages=num_stages,
                )
            )

    return configs


@triton.autotune(
    configs=_conv_transpose3d_autotune_configs(),
    key=["CIN", "COUT", "WOUT", "HOUT", "DOUT"],
)
@triton.jit
def _conv_transpose3d_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, CIN, DIN, HIN, WIN,
    COUT, DOUT, HOUT, WOUT,
    sx_n, sx_c, sx_d, sx_h, sx_w,
    sw_ci, sw_co, sw_kd, sw_kh, sw_kw,
    sy_n, sy_c, sy_d, sy_h, sy_w,
    PAD_D, PAD_H, PAD_W,
    STRIDE_D, STRIDE_H, STRIDE_W,
    NUM_CO_TILES,
    BLOCK_CO: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_CI: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid_w = tl.program_id(0)
    pid_rc = tl.program_id(1)

    tmp = pid_rc
    co_tile = tmp % NUM_CO_TILES
    tmp //= NUM_CO_TILES
    h_out = tmp % HOUT
    tmp //= HOUT
    d_out = tmp % DOUT
    n_idx = tmp // DOUT

    n_idx64 = n_idx.to(tl.int64)
    d_out64 = d_out.to(tl.int64)
    h_out64 = h_out.to(tl.int64)

    co_offsets = co_tile * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < COUT
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_out < WOUT

    acc = tl.zeros((BLOCK_W, BLOCK_CO), dtype=tl.float32)

    w_par = (w_out + PAD_W) & 1
    even_cols = w_par == 0
    odd_cols = ~even_cols

    w_in_a = (w_out + PAD_W - w_par) >> 1
    w_in_b = w_in_a - 1
    w_valid_a = (w_in_a >= 0) & (w_in_a < WIN)
    w_valid_b = (w_in_b >= 0) & (w_in_b < WIN)

    kd_base = (d_out + PAD_D) & 1
    kh_base = (h_out + PAD_H) & 1
    y_base = y_ptr + n_idx64 * sy_n + d_out64 * sy_d + h_out64 * sy_h

    for kd_sel in range(2):
        kd = kd_base + (kd_sel << 1)
        d_in = (d_out + PAD_D - kd) >> 1
        d_valid = (d_in >= 0) & (d_in < DIN)

        for kh_sel in range(2):
            kh = kh_base + (kh_sel << 1)
            h_in = (h_out + PAD_H - kh) >> 1
            h_valid = (h_in >= 0) & (h_in < HIN)
            dh_valid = d_valid & h_valid

            d_in64 = d_in.to(tl.int64)
            h_in64 = h_in.to(tl.int64)
            x_base = x_ptr + n_idx64 * sx_n + d_in64 * sx_d + h_in64 * sx_h

            col_mask_a = w_mask & w_valid_a & dh_valid
            col_mask_b = w_mask & w_valid_b & dh_valid
            col_mask_a_even = col_mask_a & even_cols
            col_mask_b_even = col_mask_b & even_cols
            col_mask_a_odd = col_mask_a & odd_cols
            col_mask_b_odd = col_mask_b & odd_cols

            for ci0 in range(0, CIN, BLOCK_CI):
                ci = ci0 + tl.arange(0, BLOCK_CI)
                ci_mask = ci < CIN
                wmask2d = ci_mask[:, None] & co_mask[None, :]

                w_ptr_base = (
                    w_ptr
                    + ci[:, None] * sw_ci
                    + co_offsets[None, :] * sw_co
                    + kd * sw_kd
                    + kh * sw_kh
                )
                x_ptr_base = x_base + ci[:, None] * sx_c

                w0 = tl.load(w_ptr_base + 0 * sw_kw, mask=wmask2d, other=0.0).to(tl.float32)
                w1 = tl.load(w_ptr_base + 1 * sw_kw, mask=wmask2d, other=0.0).to(tl.float32)
                w2 = tl.load(w_ptr_base + 2 * sw_kw, mask=wmask2d, other=0.0).to(tl.float32)
                w3 = tl.load(w_ptr_base + 3 * sw_kw, mask=wmask2d, other=0.0).to(tl.float32)

                xa_even = tl.load(
                    x_ptr_base + w_in_a[None, :] * sx_w,
                    mask=ci_mask[:, None] & col_mask_a_even[None, :],
                    other=0.0,
                ).to(tl.float32)
                xb_even = tl.load(
                    x_ptr_base + w_in_b[None, :] * sx_w,
                    mask=ci_mask[:, None] & col_mask_b_even[None, :],
                    other=0.0,
                ).to(tl.float32)
                xa_odd = tl.load(
                    x_ptr_base + w_in_a[None, :] * sx_w,
                    mask=ci_mask[:, None] & col_mask_a_odd[None, :],
                    other=0.0,
                ).to(tl.float32)
                xb_odd = tl.load(
                    x_ptr_base + w_in_b[None, :] * sx_w,
                    mask=ci_mask[:, None] & col_mask_b_odd[None, :],
                    other=0.0,
                ).to(tl.float32)

                acc += tl.sum(xa_even[:, :, None] * w0[:, None, :], axis=0)
                acc += tl.sum(xb_even[:, :, None] * w2[:, None, :], axis=0)
                acc += tl.sum(xa_odd[:, :, None] * w1[:, None, :], axis=0)
                acc += tl.sum(xb_odd[:, :, None] * w3[:, None, :], axis=0)

    bias = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    y_ptrs = y_base + co_offsets[None, :] * sy_c + w_out[:, None] * sy_w
    tl.store(y_ptrs, acc.to(tl.float16), mask=w_mask[:, None] & co_mask[None, :])


@triton.jit
def _erf_approx(x):
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    log2e = 1.4426950408889634

    sign = tl.where(x >= 0.0, 1.0, -1.0)
    ax = tl.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * t) + a4) * t) + a3) * t + a2) * t + a1
    exp_term = tl.math.exp2((-ax * ax) * log2e)
    y = 1.0 - poly * t * exp_term
    return sign * y


@triton.autotune(
    configs=_ln_gelu_autotune_configs(),
    key=["rows", "L"],
)
@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    rows, L, eps, scale,
    ROWS_PER_PROG: tl.constexpr, NORM_SIZE: tl.constexpr, grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    row_ids = row_start + tl.arange(0, ROWS_PER_PROG)
    col_ids = tl.arange(0, NORM_SIZE)

    row_mask = row_ids < rows
    col_mask = col_ids < L
    mask = row_mask[:, None] & col_mask[None, :]

    row_ids64 = row_ids.to(tl.int64)
    offs = row_ids64[:, None] * L + col_ids[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(w_ptr + col_ids, mask=col_mask, other=1.0).to(tl.float32)
    beta = tl.load(b_ptr + col_ids, mask=col_mask, other=0.0).to(tl.float32)

    l_f = tl.full((ROWS_PER_PROG,), L, tl.float32)
    mean = tl.sum(x, axis=1) / l_f
    xm = x - mean[:, None]
    var = tl.sum(xm * xm, axis=1) / l_f
    inv_std = 1.0 / tl.sqrt(var + eps)

    y = xm * inv_std[:, None]
    y = y * gamma[None, :] + beta[None, :]

    z = y * 0.7071067811865476
    erfz = _erf_approx(z)
    out = (0.5 * y * (1.0 + erfz)) * scale

    tl.store(y_ptr + offs, out.to(tl.float16), mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


def conv_transpose3d_bias(
    x,
    weight,
    bias,
    stride=(2, 2, 2),
    padding=(1, 1, 1),
    output_padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
):
    assert x.device.type == "xpu"
    assert weight.device.type == "xpu"
    assert bias.device.type == "xpu"
    assert x.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert bias.dtype == torch.float16
    assert stride == (2, 2, 2) and padding == (1, 1, 1)
    assert dilation == (1, 1, 1) and output_padding == (0, 0, 0) and groups == 1
    return F.conv_transpose3d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


def ln_gelu_scale(x, weight, bias, eps=1e-5, scale=1.0):
    assert x.device.type == "xpu"
    assert x.dtype == torch.float16
    assert x.is_contiguous()

    L = x.shape[-1]
    assert weight.numel() == L and bias.numel() == L
    assert weight.device.type == "xpu" and bias.device.type == "xpu"

    rows = x.numel() // L
    out = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(rows, META["ROWS_PER_PROG"]),)
    _ln_gelu_scale_kernel[grid](
        x, weight, bias, out,
        rows, L, eps, float(scale),
        NORM_SIZE=L,
    )
    return out


def kernel_function(
    x,
    conv_weight,
    conv_bias,
    ln_weight,
    ln_bias,
    stride=(2, 2, 2),
    padding=(1, 1, 1),
    eps=1e-5,
    scale=1.0,
):
    assert x.device.type == "xpu"
    assert conv_weight.device.type == "xpu"
    assert conv_bias.device.type == "xpu"
    assert ln_weight.device.type == "xpu"
    assert ln_bias.device.type == "xpu"

    if x.dtype != torch.float16 or not x.is_contiguous():
        x = x.to("xpu", dtype=torch.float16).contiguous()

    y1 = conv_transpose3d_bias(
        x,
        conv_weight,
        conv_bias,
        stride=stride,
        padding=padding,
    )
    y2 = ln_gelu_scale(
        y1.contiguous(),
        ln_weight,
        ln_bias,
        eps=eps,
        scale=scale,
    )
    return y2


batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        eps=1e-5,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        self.scale = 1.0
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor
        self._xpu_params_prepared = False

    def _prepare_xpu_params(self):
        if not self._xpu_params_prepared:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.data = self.conv_transpose.bias.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()
            self.layer_norm.weight.data = self.layer_norm.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
            self.layer_norm.bias.data = self.layer_norm.bias.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
            self._xpu_params_prepared = True

    def forward(self, x):
        self._prepare_xpu_params()

        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to("xpu", dtype=torch.float16).contiguous()

        s = self.stride
        p = self.padding
        stride_t = (s, s, s) if isinstance(s, int) else tuple(s)
        padding_t = (p, p, p) if isinstance(p, int) else tuple(p)

        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.layer_norm.weight,
            self.layer_norm.bias,
            stride=stride_t,
            padding=padding_t,
            eps=self.eps,
            scale=self.scale,
        )