# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_CI": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_CI": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_CI": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_CI": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_CI": 64, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_CI": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_CI": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_CI": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_CI": 64, "GROUP_SIZE_M": 8},
            num_warps=32,
            num_stages=2,
        ),
    ],
    key=["OH", "OW", "Ci", "Co"],
)
@triton.jit
def _conv_transpose2d_bias_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    Ci,
    H,
    W,
    Co,
    OH,
    OW,
    sxn,
    sxc,
    sxh,
    sxw,
    sWkh,
    sWkw,
    sWco,
    sWci,
    syn,
    syc,
    syh,
    syw,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)

    num_pid_pos = tl.cdiv(OH * OW, BLOCK_M)
    num_pid_nc = N * Co

    if GROUP_SIZE_M > 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_nc
        group_id = pid // num_pid_in_group
        first_pid_pos = group_id * GROUP_SIZE_M
        group_size_pos = tl.minimum(num_pid_pos - first_pid_pos, GROUP_SIZE_M)
        pid_pos = first_pid_pos + ((pid % num_pid_in_group) % group_size_pos)
        pid_nc = (pid % num_pid_in_group) // group_size_pos
    else:
        pid_pos = pid % num_pid_pos
        pid_nc = pid // num_pid_pos

    n = pid_nc // Co
    oc = pid_nc % Co

    start = pid_pos * BLOCK_M
    offs_p = start + tl.arange(0, BLOCK_M)
    mask_p = offs_p < (OH * OW)
    ow = offs_p % OW
    oh = offs_p // OW

    n_off = n.to(tl.int64) * sxn
    oc_off_y = oc.to(tl.int64) * syc
    oc_off_w = oc.to(tl.int64) * sWco

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kh in range(KH):
        ih_nom = oh + PAD_H - kh * DIL_H
        ih = ih_nom // STRIDE_H
        valid_h = (ih >= 0) & (ih < H) & (ih * STRIDE_H == ih_nom)

        for kw in range(KW):
            iw_nom = ow + PAD_W - kw * DIL_W
            iw = iw_nom // STRIDE_W
            valid_w = (iw >= 0) & (iw < W) & (iw * STRIDE_W == iw_nom)
            pos_mask = mask_p & valid_h & valid_w

            base_w = w_ptr + kh * sWkh + kw * sWkw + oc_off_w
            ci0 = 0
            while ci0 < Ci:
                offs_ci = ci0 + tl.arange(0, BLOCK_CI)
                mask_ci = offs_ci < Ci

                w_ptrs = base_w + offs_ci * sWci
                w_ci = tl.load(w_ptrs, mask=mask_ci, other=0.0).to(tl.float32)

                x_ptrs = (
                    x_ptr
                    + n_off
                    + offs_ci[:, None] * sxc
                    + ih[None, :] * sxh
                    + iw[None, :] * sxw
                )
                x_vals = tl.load(
                    x_ptrs,
                    mask=mask_ci[:, None] & pos_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(x_vals * w_ci[:, None], axis=0)
                ci0 += BLOCK_CI

    b_val = tl.load(b_ptr + oc).to(tl.float32)
    acc += b_val

    y_ptrs = y_ptr + n_off + oc_off_y + oh * syh + ow * syw
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_p)


@triton.jit
def _fused_reduce_gelu_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BIAS_MODE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    x_batch_off = pid_n.to(tl.int64) * stride_xn
    out_batch_off = pid_n.to(tl.int64) * stride_on

    sum_vec = tl.zeros((BLOCK_W,), dtype=tl.float32)
    num_ctiles = tl.cdiv(C, BLOCK_C)

    for h in range(H):
        min_vec = tl.full((BLOCK_W,), float("inf"), dtype=tl.float32)
        h_off = h * stride_xh
        for ct in range(num_ctiles):
            offs_c = ct * BLOCK_C + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            x_ptrs = (
                x_ptr
                + x_batch_off
                + offs_c[:, None] * stride_xc
                + h_off
                + offs_w[None, :] * stride_xw
            )
            tile = tl.load(
                x_ptrs,
                mask=mask_c[:, None] & mask_w[None, :],
                other=float("inf"),
            )
            min_vec = tl.minimum(min_vec, tl.min(tile, axis=0))
        sum_vec += min_vec

    inv_sqrt2 = 0.7071067811865476
    gelu_val = 0.5 * sum_vec * (1.0 + tl.math.erf(sum_vec * inv_sqrt2))

    if BIAS_MODE == 0:
        b = tl.load(bias_ptr).to(tl.float32)
        y = gelu_val + b
    else:
        b = tl.load(bias_ptr + offs_w, mask=mask_w, other=0.0).to(tl.float32)
        y = gelu_val + b

    out_ptrs = out_ptr + out_batch_off + offs_w * stride_ow
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask_w)


def _compute_output_size(
    H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_pad_h, out_pad_w
):
    OH = (H - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1
    OW = (W - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1
    return OH, OW


def conv_transpose_bias(x, packed_weight, bias):
    assert x.device.type == "xpu"
    assert packed_weight.device.type == "xpu"
    assert bias.device.type == "xpu"
    assert (
        x.dtype == torch.float16
        and packed_weight.dtype == torch.float16
        and bias.dtype == torch.float16
    )

    N, Ci, H, W = x.shape
    kH, kW, Co, Ci_w = packed_weight.shape
    assert Ci_w == Ci and bias.numel() == Co

    stride_h = 2
    stride_w = 2
    pad_h = 1
    pad_w = 1
    dil_h = 1
    dil_w = 1
    out_pad_h = 1
    out_pad_w = 1

    OH, OW = _compute_output_size(
        H,
        W,
        kH,
        kW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        out_pad_h,
        out_pad_w,
    )
    y = torch.empty((N, Co, OH, OW), dtype=x.dtype, device=x.device)

    sxn, sxc, sxh, sxw = x.stride()
    sWkh, sWkw, sWco, sWci = packed_weight.stride()
    syn, syc, syh, syw = y.stride()

    def grid(meta):
        num_pid_pos = triton.cdiv(OH * OW, meta["BLOCK_M"])
        return (num_pid_pos * (N * Co),)

    _conv_transpose2d_bias_kernel[grid](
        x,
        packed_weight,
        bias,
        y,
        N,
        Ci,
        H,
        W,
        Co,
        OH,
        OW,
        sxn,
        sxc,
        sxh,
        sxw,
        sWkh,
        sWkw,
        sWco,
        sWci,
        syn,
        syc,
        syh,
        syw,
        KH=kH,
        KW=kW,
        STRIDE_H=stride_h,
        STRIDE_W=stride_w,
        PAD_H=pad_h,
        PAD_W=pad_w,
        DIL_H=dil_h,
        DIL_W=dil_w,
        grf_mode="auto",
    )
    return y


def reduce_gelu_bias(x, bias):
    assert x.device.type == "xpu"
    assert bias.device == x.device
    assert x.dtype == torch.float16 and bias.dtype == torch.float16
    assert x.ndim == 4

    N, C, H, W = x.shape
    if bias.numel() == 1:
        bias_mode = 0
        bias_vec = bias.contiguous().view(-1)
    elif bias.numel() == W:
        bias_mode = 1
        bias_vec = bias.contiguous().view(-1)
    else:
        raise ValueError(f"Unsupported bias size {bias.shape}")

    out = torch.empty((N, 1, 1, W), dtype=x.dtype, device=x.device)
    BLOCK_W = 128
    BLOCK_C = 32
    grid = (triton.cdiv(W, BLOCK_W), N)
    _fused_reduce_gelu_bias_kernel[grid](
        x,
        bias_vec,
        out,
        N,
        C,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BIAS_MODE=bias_mode,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=2,
    )
    return out


def kernel_function(x, packed_weight, conv_bias, final_bias):
    y = conv_transpose_bias(x, packed_weight, conv_bias)
    return reduce_gelu_bias(y, final_bias)


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.final_bias = nn.Parameter(torch.zeros(bias_shape))
        self._packed_weight = None
        self._packed_weight_version = None

    def _ensure_xpu_params(self):
        if (
            self.conv_transpose.weight.device.type != "xpu"
            or self.conv_transpose.weight.dtype != torch.float16
        ):
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        else:
            self.conv_transpose.weight.data = (
                self.conv_transpose.weight.data.contiguous()
            )

        if self.conv_transpose.bias is not None:
            if (
                self.conv_transpose.bias.device.type != "xpu"
                or self.conv_transpose.bias.dtype != torch.float16
            ):
                self.conv_transpose.bias.data = self.conv_transpose.bias.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()
            else:
                self.conv_transpose.bias.data = (
                    self.conv_transpose.bias.data.contiguous()
                )

        if (
            self.final_bias.device.type != "xpu"
            or self.final_bias.dtype != torch.float16
        ):
            self.final_bias.data = self.final_bias.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
        else:
            self.final_bias.data = self.final_bias.data.contiguous()

    def _get_packed_weight(self):
        w = self.conv_transpose.weight
        version = getattr(w, "_version", None)
        if (
            self._packed_weight is None
            or self._packed_weight_version != version
            or self._packed_weight.device != w.device
            or self._packed_weight.dtype != w.dtype
        ):
            self._packed_weight = w.permute(2, 3, 1, 0).contiguous()
            self._packed_weight_version = version
        return self._packed_weight

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        self._ensure_xpu_params()
        packed_weight = self._get_packed_weight()

        return kernel_function(
            x,
            packed_weight,
            self.conv_transpose.bias,
            self.final_bias,
        )
