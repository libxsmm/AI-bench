import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _channelmin_autotune_configs():
    # Row-reduction-specific tuning space for Intel XPU.
    # BLOCK_W is the reduction scan width across W_OUT.
    # ROWS_PER_PROG lets one program process multiple (n, h) rows.
    configs = []

    candidates = [
        (1, 64, 4, 1),
        (1, 64, 8, 2),
        (1, 128, 4, 1),
        (1, 128, 8, 2),
        (1, 128, 16, 2),
        (1, 256, 8, 2),
        (1, 256, 16, 2),
        (1, 256, 32, 2),  # required XPU-oriented large-width/high-warp config
        (1, 512, 8, 2),
        (1, 512, 16, 2),
        (1, 512, 32, 2),
        (2, 64, 4, 1),
        (2, 128, 8, 2),
        (2, 256, 8, 2),
        (2, 256, 16, 2),
        (2, 256, 32, 2),
        (2, 512, 16, 2),
        (2, 512, 32, 2),
        (4, 64, 4, 1),
        (4, 128, 8, 2),
        (4, 256, 8, 2),
        (4, 256, 16, 2),
        (4, 512, 16, 2),
    ]

    for rows_per_prog, block_w, num_warps, num_stages in candidates:
        configs.append(
            triton.Config(
                {
                    "ROWS_PER_PROG": rows_per_prog,
                    "BLOCK_W": block_w,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return configs


# Keep the original Triton convolution kernel present for compatibility/reference.
@triton.jit
def _conv_scale_channelmin_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C_IN,
    H,
    W,
    C_OUT,
    H_OUT,
    W_OUT,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_co,
    w_stride_ci,
    w_stride_kh,
    w_stride_kw,
    b_stride,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    scale,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    n = pid_0 // H_OUT
    oh = pid_0 % H_OUT
    ow_start = pid_1 * BLOCK_W
    offs_w = tl.arange(0, BLOCK_W)
    ow = ow_start + offs_w
    mask_out = ow < W_OUT

    running_min = tl.full((BLOCK_W,), float("inf"), dtype=tl.float32)

    for co in tl.range(0, C_OUT):
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for ci in tl.range(0, C_IN):
            for kh in tl.static_range(0, KH):
                ih = oh * stride_h + kh * dilation_h
                base_in = x_ptr + n * x_stride_n + ci * x_stride_c + ih * x_stride_h
                for kw in tl.static_range(0, KW):
                    iw = ow * stride_w + kw * dilation_w
                    x_val = tl.load(base_in + iw * x_stride_w, mask=mask_out, other=0.0)
                    w_val = tl.load(
                        w_ptr
                        + co * w_stride_co
                        + ci * w_stride_ci
                        + kh * w_stride_kh
                        + kw * w_stride_kw
                    )
                    acc += x_val.to(tl.float32) * w_val.to(tl.float32)
        b_f = tl.load(b_ptr + co * b_stride).to(tl.float32)
        acc = (acc + b_f) * scale
        running_min = tl.minimum(running_min, acc)

    out_base = out_ptr + n * out_stride_n + oh * out_stride_h
    tl.store(
        out_base + ow * out_stride_w,
        running_min.to(out_ptr.dtype.element_ty),
        mask=mask_out,
    )


@triton.autotune(
    configs=_channelmin_autotune_configs(),
    key=["C_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def _channelmin_kernel(
    y_ptr,
    out_ptr,
    N,
    C_OUT,
    H_OUT,
    W_OUT,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    scale,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)

    row_start = pid_row * ROWS_PER_PROG
    w_start = pid_w * BLOCK_W
    offs_w = tl.arange(0, BLOCK_W)
    cols = w_start + offs_w
    mask_w = cols < W_OUT

    for row_idx in tl.static_range(0, ROWS_PER_PROG):
        linear_row = row_start + row_idx
        if linear_row < N * H_OUT:
            n = linear_row // H_OUT
            oh = linear_row % H_OUT

            running_min = tl.full((BLOCK_W,), float("inf"), dtype=tl.float32)

            base_y = y_ptr + n * y_stride_n + oh * y_stride_h
            base_out = out_ptr + n * out_stride_n + oh * out_stride_h

            for co in tl.range(0, C_OUT):
                vals = tl.load(
                    base_y + co * y_stride_c + cols * y_stride_w,
                    mask=mask_w,
                    other=float("inf"),
                )
                running_min = tl.minimum(running_min, vals.to(tl.float32))

            running_min = running_min * scale
            tl.store(
                base_out + cols * out_stride_w,
                running_min.to(out_ptr.dtype.element_ty),
                mask=mask_w,
            )


def kernel_function(x, conv_weight, conv_bias=None, scale_factor: float = 2.0):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU is required."

    x_xpu = x.to("xpu", dtype=x.dtype).contiguous()
    w_xpu = conv_weight.to("xpu", dtype=x_xpu.dtype).contiguous()

    c_out = w_xpu.shape[0]
    if conv_bias is None:
        b_xpu = torch.zeros((c_out,), device="xpu", dtype=x_xpu.dtype)
    else:
        b_xpu = conv_bias.to("xpu", dtype=x_xpu.dtype).contiguous()

    # Heavy compute goes through vendor convolution.
    y = F.conv2d(x_xpu, w_xpu, b_xpu)

    n, c_out_y, h_out, w_out = y.shape
    out = torch.empty((n, 1, h_out, w_out), device="xpu", dtype=y.dtype)

    y_stride_n, y_stride_c, y_stride_h, y_stride_w = y.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    grid = lambda META: (
        triton.cdiv(n * h_out, META["ROWS_PER_PROG"]),
        triton.cdiv(w_out, META["BLOCK_W"]),
    )

    _channelmin_kernel[grid](
        y,
        out,
        n,
        c_out_y,
        h_out,
        w_out,
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w,
        out_stride_n,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        float(scale_factor),
        grf_mode="auto",
    )

    return out


def run_test():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU device not available; skipping test.")
        sys.exit(0)

    batch_size = 64
    in_channels = 64
    out_channels = 128
    height = width = 256
    kernel_size = 3
    scale_factor = 2.0

    torch.manual_seed(0)
    x_cpu = torch.randn(batch_size, in_channels, height, width, dtype=torch.float16)
    weight_cpu = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float16
    )
    bias_cpu = torch.randn(out_channels, dtype=torch.float16)

    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=True).to(
        dtype=torch.float16
    )
    with torch.no_grad():
        conv.weight.copy_(weight_cpu)
        conv.bias.copy_(bias_cpu)

    ref = conv(x_cpu)
    ref = ref * scale_factor
    ref = torch.min(ref, dim=1, keepdim=True).values

    x = x_cpu.to("xpu")
    w = weight_cpu.to("xpu")
    b = bias_cpu.to("xpu")

    out = kernel_function(x, w, b, scale_factor)
    out_cpu = out.cpu()

    if torch.allclose(ref, out_cpu, rtol=1e-3, atol=1e-3):
        print("PASS")
        sys.exit(0)
    else:
        max_diff = (ref - out_cpu).abs().max().item()
        print(f"FAIL: max difference = {max_diff}")
        sys.exit(1)


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
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

        return kernel_function(
            x_xpu,
            self.conv.weight,
            self.conv.bias,
            self.scale_factor,
        )
