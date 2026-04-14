# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _fused_add_mul_hardswish_configs():
    # Elementwise kernel: tune BLOCK_SIZE / warps / stages only.
    # grf_mode must NOT appear in triton.Config(); it remains a compiler option
    # declared in the signature and selected at launch.
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
    ]


# --------------------------------------------------------
# Baseline kernels kept for reference/compatibility
# --------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OC": 32, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OC": 32, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OC": 16, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OC": 64, "BLOCK_W": 32}, num_warps=8, num_stages=2),
    ],
    key=["Cout", "Wout"],
)
@triton.jit
def _deconv3d_bias_add_kernel(
    x_ptr,
    add_ptr,
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
    sx_n,
    sx_c,
    sx_d,
    sx_h,
    sx_w,
    sy_n,
    sy_c,
    sy_d,
    sy_h,
    sy_w,
    sw_ic,
    sw_oc,
    sw_kd,
    sw_kh,
    sw_kw,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    dh = Dout * Hout
    n = pid_m // dh
    rem = pid_m % dh
    od = rem // Hout
    oh = rem % Hout

    oc_start = pid_oc * BLOCK_OC
    oc_offs = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < Cout

    ow_start = pid_w * BLOCK_W
    ow_offs = ow_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < Wout

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    for kd in range(KD):
        num_d = od + pad_d - kd * dil_d
        divd_ok = (num_d % stride_d) == 0
        id = num_d // stride_d
        id_in = (id >= 0) & (id < Din)
        dh_ok = divd_ok & id_in

        for kh in range(KH):
            num_h = oh + pad_h - kh * dil_h
            divh_ok = (num_h % stride_h) == 0
            ih = num_h // stride_h
            ih_in = (ih >= 0) & (ih < Hin)
            dhh_ok = dh_ok & divh_ok & ih_in

            for kw in range(KW):
                num_w = ow_offs + pad_w - kw * dil_w
                divw_ok = (num_w % stride_w) == 0
                iw = num_w // stride_w
                iw_in = (iw >= 0) & (iw < Win)
                mask_w = ow_mask & divw_ok & iw_in
                mask_x = mask_w & dhh_ok

                base_x = n * sx_n + id * sx_d + ih * sx_h
                x_ptrs = x_ptr + base_x + iw * sx_w

                for ic in range(Cin):
                    x_vals = tl.load(x_ptrs + ic * sx_c, mask=mask_x, other=0.0)
                    w_ptrs = (
                        w_ptr
                        + ic * sw_ic
                        + oc_offs * sw_oc
                        + kd * sw_kd
                        + kh * sw_kh
                        + kw * sw_kw
                    )
                    w_vec = tl.load(w_ptrs, mask=oc_mask, other=0.0)
                    acc += w_vec[:, None] * x_vals[None, :]

    b_vec = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + b_vec[:, None]

    add_ptrs = (
        add_ptr
        + n * sy_n
        + oc_offs[:, None] * sy_c
        + od * sy_d
        + oh * sy_h
        + ow_offs[None, :] * sy_w
    )
    add_mask = oc_mask[:, None] & ow_mask[None, :]
    add_vals = tl.load(add_ptrs, mask=add_mask, other=0.0)
    acc = acc + add_vals

    y_ptrs = (
        y_ptr
        + n * sy_n
        + oc_offs[:, None] * sy_c
        + od * sy_d
        + oh * sy_h
        + ow_offs[None, :] * sy_w
    )
    tl.store(y_ptrs, acc, mask=add_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _fused_mul_hardswish_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    t = x + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    y = (x * x) * t * (1.0 / 6.0)
    tl.store(y_ptr + offsets, y, mask=mask)


# --------------------------------------------------------
# Existing optimized epilogue-only Triton kernel retained
# for compatibility with prior stages.
# Intel XPU: grf_mode is a compiler option exposed as
# constexpr arg, not part of triton.Config().
# --------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _mul_hardswish_inplace_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = x + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    y = x * x * t * (1.0 / 6.0)
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


# --------------------------------------------------------
# Fused epilogue kernel:
# y = (z + add) * hardswish(z + add)
# Keeps vendor conv_transpose3d intact, but removes the
# separate materialized add kernel/op.
# Intel XPU: grf_mode is a compiler option exposed as
# constexpr arg, not part of triton.Config().
# --------------------------------------------------------
@triton.autotune(
    configs=_fused_add_mul_hardswish_configs(),
    key=["N"],
)
@triton.jit
def _fused_add_mul_hardswish_kernel(
    x_ptr,
    add_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    add = tl.load(add_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = x + add

    t = v + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    y = v * v * t * (1.0 / 6.0)

    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


# --------------------------------------------------------
# Top-level wrapper
# --------------------------------------------------------
def kernel_function(
    x: torch.Tensor,
    add_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(add_in, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and isinstance(bias, torch.Tensor)
    ):
        raise TypeError("All inputs must be torch.Tensors")

    if x.device.type != "xpu":
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.to(dtype=torch.float16).contiguous()

    if add_in.device.type != "xpu":
        add_xpu = add_in.to("xpu", dtype=torch.float16).contiguous()
    else:
        add_xpu = add_in.to(dtype=torch.float16).contiguous()

    if weight.device.type != "xpu":
        w_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
    else:
        w_xpu = weight.to(dtype=torch.float16).contiguous()

    if bias.device.type != "xpu":
        b_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        b_xpu = bias.to(dtype=torch.float16).contiguous()

    if x_xpu.ndim != 5 or add_xpu.ndim != 5:
        raise ValueError("x and add_in must be 5D (N,C,D,H,W)")

    N, Cin, Din, Hin, Win = x_xpu.shape
    N2, Cout, Dout, Hout, Wout = add_xpu.shape
    if N2 != N:
        raise ValueError("Batch size mismatch")

    if w_xpu.ndim != 5:
        raise ValueError("weight must be 5D [Cin, Cout, Kd, Kh, Kw]")
    Cin_w, Cout_w, Kd, Kh, Kw = w_xpu.shape
    if Cin_w != Cin or Cout_w != Cout:
        raise ValueError("Weight channels must match x and add_in")

    if b_xpu.ndim != 1 or b_xpu.shape[0] != Cout:
        raise ValueError("bias must be 1D of length Cout")

    z = torch.nn.functional.conv_transpose3d(
        x_xpu,
        w_xpu,
        bias=b_xpu,
        stride=2,
        padding=1,
        output_padding=1,
        dilation=1,
    )

    if z.shape != add_xpu.shape:
        raise ValueError("conv_transpose3d output shape must match add_in shape")

    y = torch.empty_like(z)
    n_elems = z.numel()

    def grid(meta):
        return (triton.cdiv(n_elems, meta["BLOCK_SIZE"]),)

    _fused_add_mul_hardswish_kernel[grid](z, add_xpu, y, n_elems, grf_mode="auto")
    return y


# --------------------------------------------------------
# Self-test
# --------------------------------------------------------
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W),
        torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride),
    ]


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
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=1,
            output_padding=output_padding,
        )
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_shape = bias_shape

    def forward(self, x, add_input):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if add_input.device.type != "xpu" or add_input.dtype != torch.float16:
            add_input = add_input.to("xpu", dtype=torch.float16)

        w = self.conv_transpose.weight
        b = self.conv_transpose.bias

        if w.device.type != "xpu" or w.dtype != torch.float16:
            w = w.to("xpu", dtype=torch.float16).contiguous()
        else:
            w = w.contiguous()

        if b.device.type != "xpu" or b.dtype != torch.float16:
            b = b.to("xpu", dtype=torch.float16).contiguous()
        else:
            b = b.contiguous()

        return kernel_function(x, add_input, w, b)
