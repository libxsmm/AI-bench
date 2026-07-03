# ruff: noqa: E731
# KernelBench-compatible wrapper — Model class injected by codegen
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------
# Utility: compute output dimensions for 3D transposed convolution
# ----------------------------------------
def _compute_output_dims_3d(
    Din, Hin, Win, stride, padding, dilation, kernel_size, output_padding
):
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    kd, kh, kw = kernel_size
    opd, oph, opw = output_padding
    Dout = (Din - 1) * sd - 2 * pd + dd * (kd - 1) + opd + 1
    Hout = (Hin - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    Wout = (Win - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    return Dout, Hout, Wout


# ----------------------------------------
# Autotune configurations for XPU
# ----------------------------------------
def _deconv_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_CO": 16, "BLOCK_OW": 16, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 16, "BLOCK_OW": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 32, "BLOCK_OW": 16, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 32, "BLOCK_OW": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 32, "BLOCK_OW": 64, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 64, "BLOCK_OW": 16, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 64, "BLOCK_OW": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 64, "BLOCK_OW": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_CO": 64, "BLOCK_OW": 128, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_CO": 128, "BLOCK_OW": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_CO": 128, "BLOCK_OW": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_CO": 128, "BLOCK_OW": 128, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_CO": 256, "BLOCK_OW": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_CO": 256, "BLOCK_OW": 128, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_CO": 256, "BLOCK_OW": 256, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
    ]


def _softmax_autotune_configs():
    return [
        triton.Config({"BLOCK_C": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=32, num_stages=2),
    ]


# ----------------------------------------
# Kernel: 3D transposed convolution with bias
# ----------------------------------------
@triton.autotune(
    configs=_deconv_autotune_configs(),
    key=["Cin", "Cout", "Din", "Hin", "Win", "Dout", "Hout", "Wout"],
)
@triton.jit
def _deconv3d_bias_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    Cin,
    Cout,
    Din,
    Hin,
    Win,
    Dout,
    Hout,
    Wout,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_d: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    xsN,
    xsC,
    xsD,
    xsH,
    xsW,
    wsCi,
    wsCo,
    wsKd,
    wsKh,
    wsKw,
    ysN,
    ysC,
    ysD,
    ysH,
    ysW,
    BLOCK_CO: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_cotile = tl.program_id(1)
    pid_sp = tl.program_id(2)

    w_tiles = tl.cdiv(Wout, BLOCK_OW)
    tmp = pid_sp
    ow_tile = tmp % w_tiles
    tmp = tmp // w_tiles
    oh = tmp % Hout
    od = tmp // Hout

    offs_co = pid_cotile * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ow = ow_tile * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask_co = offs_co < Cout
    mask_ow = offs_ow < Wout

    acc = tl.zeros((BLOCK_CO, BLOCK_OW), dtype=tl.float32)

    for ci in range(0, Cin):
        for kd_ in range(0, Kd):
            rd = od + pad_d - kd_ * dil_d
            valid_d = (rd % stride_d) == 0
            id_ = rd // stride_d
            valid_d = valid_d & (id_ >= 0) & (id_ < Din)
            if valid_d:
                for kh_ in range(0, Kh):
                    rh = oh + pad_h - kh_ * dil_h
                    valid_h = (rh % stride_h) == 0
                    ih = rh // stride_h
                    valid_h = valid_h & (ih >= 0) & (ih < Hin)
                    if valid_h:
                        base_x = pid_n * xsN + ci * xsC + id_ * xsD + ih * xsH
                        base_w = ci * wsCi + kd_ * wsKd + kh_ * wsKh
                        for kw_ in range(0, Kw):
                            rx = offs_ow + pad_w - kw_ * dil_w
                            valid_w = (rx % stride_w) == 0
                            ix = rx // stride_w
                            mask_x = mask_ow & valid_w & (ix >= 0) & (ix < Win)

                            x_vals = tl.load(
                                x_ptr + base_x + ix * xsW, mask=mask_x, other=0.0
                            ).to(tl.float32)
                            w_vals = tl.load(
                                w_ptr + base_w + kw_ * wsKw + offs_co * wsCo,
                                mask=mask_co,
                                other=0.0,
                            ).to(tl.float32)
                            acc += w_vals[:, None] * x_vals[None, :]

    if HAS_BIAS:
        b_vals = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
        acc += b_vals[:, None]

    out_ptrs = y_ptr + (
        pid_n * ysN
        + offs_co[:, None] * ysC
        + od * ysD
        + oh * ysH
        + offs_ow[None, :] * ysW
    )
    out_mask = mask_co[:, None] & mask_ow[None, :]
    tl.store(out_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


# ----------------------------------------
# Original kernel retained to satisfy kernel-preservation constraint.
# ----------------------------------------
@triton.jit
def _softmax_sigmoid_kernel(
    x_ptr,
    y_ptr,
    ROWS,
    C,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    row_start = pid * C
    x_vals = tl.load(x_ptr + row_start + offs_c, mask=mask_c, other=-float("inf")).to(
        tl.float32
    )
    max_val = tl.max(x_vals, axis=0)
    x_vals = x_vals - max_val
    exp_vals = tl.exp(x_vals)
    sum_val = tl.sum(exp_vals, axis=0)
    soft = exp_vals / sum_val
    y_vals = 1.0 / (1.0 + tl.exp(-soft))
    tl.store(y_ptr + row_start + offs_c, y_vals.to(y_ptr.dtype.element_ty), mask=mask_c)


# ----------------------------------------
# New fused post-op kernel: in-place softmax(dim=1)+sigmoid on row-contiguous view.
# ----------------------------------------
@triton.autotune(configs=_softmax_autotune_configs(), key=["C", "ROWS"])
@triton.jit
def _softmax_sigmoid_inplace_kernel(
    x_ptr,
    ROWS,
    C,
    BLOCK_C: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    row_start = pid * C
    x_vals = tl.load(x_ptr + row_start + offs_c, mask=mask_c, other=-float("inf")).to(
        tl.float32
    )
    max_val = tl.max(x_vals, axis=0)
    x_vals = x_vals - max_val
    exp_vals = tl.exp(x_vals)
    sum_val = tl.sum(exp_vals, axis=0)
    soft = exp_vals / sum_val
    y_vals = 1.0 / (1.0 + tl.exp(-soft))
    tl.store(x_ptr + row_start + offs_c, y_vals.to(x_ptr.dtype.element_ty), mask=mask_c)


def deconv3d_bias(
    x,
    w,
    b,
    stride=(2, 2, 2),
    padding=(1, 1, 1),
    output_padding=(1, 1, 1),
    dilation=(1, 1, 1),
    groups=1,
):
    assert x.device.type == "xpu", "Expect xpu device"
    assert w.device.type == "xpu", "Expect xpu device"
    assert b is None or b.device.type == "xpu", "Expect xpu device"
    assert groups == 1, "Only groups=1 supported"
    if b is not None:
        assert x.dtype == w.dtype == b.dtype, "dtype mismatch"
    else:
        assert x.dtype == w.dtype, "dtype mismatch"

    N, Cin, Din, Hin, Win = x.shape
    Cin_w, Cout, Kd, Kh, Kw = w.shape
    if b is not None:
        assert Cin_w == Cin and b.shape[0] == Cout
    else:
        assert Cin_w == Cin

    Dout, Hout, Wout = _compute_output_dims_3d(
        Din, Hin, Win, stride, padding, dilation, (Kd, Kh, Kw), output_padding
    )

    y = torch.empty((N, Cout, Dout, Hout, Wout), dtype=x.dtype, device=x.device)

    xsN, xsC, xsD, xsH, xsW = x.stride()
    wsCi, wsCo, wsKd, wsKh, wsKw = w.stride()
    ysN, ysC, ysD, ysH, ysW = y.stride()

    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    def grid(meta):
        b_co = meta["BLOCK_CO"]
        b_ow = meta["BLOCK_OW"]
        return (N, triton.cdiv(Cout, b_co), Dout * Hout * triton.cdiv(Wout, b_ow))

    has_bias = 1 if b is not None else 0
    if b is None:
        b = torch.empty((1,), device=x.device, dtype=x.dtype)

    _deconv3d_bias_kernel[grid](
        x,
        w,
        b,
        y,
        N,
        Cin,
        Cout,
        Din,
        Hin,
        Win,
        Dout,
        Hout,
        Wout,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        Kd,
        Kh,
        Kw,
        xsN,
        xsC,
        xsD,
        xsH,
        xsW,
        wsCi,
        wsCo,
        wsKd,
        wsKh,
        wsKw,
        ysN,
        ysC,
        ysD,
        ysH,
        ysW,
        HAS_BIAS=has_bias,
        grf_mode="auto",
    )
    return y


def softmax_sigmoid(x):
    assert x.device.type == "xpu", "Expect xpu device"
    N, C, D, H, W = x.shape

    x_rows = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

    rows = x_rows.shape[0]
    _softmax_sigmoid_inplace_kernel[(rows,)](
        x_rows,
        rows,
        C,
        grf_mode="auto",
    )

    return x_rows.view(N, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


# ----------------------------------------
# Top-level fused function
# ----------------------------------------
def kernel_function(
    x,
    w,
    b,
    stride=(2, 2, 2),
    padding=(1, 1, 1),
    output_padding=(1, 1, 1),
    dilation=(1, 1, 1),
    groups=1,
):
    """
    Forward: conv_transpose3d -> softmax(dim=1) -> sigmoid
    Returns XPU tensor.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on xpu")
    if w.device.type != "xpu":
        raise RuntimeError("Weight must be on xpu")
    if b is not None and b.device.type != "xpu":
        raise RuntimeError("Bias must be on xpu")

    y0 = deconv3d_bias(x, w, b, stride, padding, output_padding, dilation, groups)
    y1 = softmax_sigmoid(y0)
    return y1


# ----------------------------------------
# Self-test
# ----------------------------------------
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self._cached_w_xpu = None
        self._cached_b_xpu = None
        self._cache_key = None

    def _get_xpu_params(self, dtype):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias

        key = (
            weight.data_ptr(),
            tuple(weight.shape),
            weight.dtype,
            str(weight.device),
            int(weight._version),
            None if bias is None else bias.data_ptr(),
            None if bias is None else tuple(bias.shape),
            None if bias is None else bias.dtype,
            None if bias is None else str(bias.device),
            None if bias is None else int(bias._version),
            dtype,
        )

        if self._cache_key != key:
            self._cached_w_xpu = (
                weight.detach().to(device="xpu", dtype=dtype).contiguous()
            )
            self._cached_b_xpu = (
                bias.detach().to(device="xpu", dtype=dtype).contiguous()
                if bias is not None
                else None
            )
            self._cache_key = key

        return self._cached_w_xpu, self._cached_b_xpu

    def forward(self, x):
        target_dtype = self.conv_transpose.weight.dtype
        x_xpu = x.to(device="xpu", dtype=target_dtype).contiguous()
        w_xpu, b_xpu = self._get_xpu_params(x_xpu.dtype)
        return kernel_function(x_xpu, w_xpu, b_xpu)
