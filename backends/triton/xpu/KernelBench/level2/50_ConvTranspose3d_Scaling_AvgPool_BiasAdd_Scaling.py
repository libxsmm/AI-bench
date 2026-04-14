# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1_val = 0.5
scale2_val = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1_val, scale2_val, bias_shape]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float16)]


@triton.jit
def _conv_transpose3d_mul1_kernel(
    x_ptr, w_ptr, b_ptr, scale1_ptr, y_ptr,
    N, C_IN, D_IN, H_IN, W_IN,
    C_OUT, KD, KH, KW,
    D_OUT, H_OUT, W_OUT,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_ci, w_stride_co, w_stride_kd, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    BLOCK_W: tl.constexpr, BLOCK_OC: tl.constexpr, NUM_WARPS: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_ndh = tl.program_id(1)
    pid_oc = tl.program_id(2)

    oh = pid_ndh % H_OUT
    tmp = pid_ndh // H_OUT
    od = tmp % D_OUT
    n = tmp // D_OUT

    oc_start = pid_oc * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < C_OUT

    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    w_mask = offs_w < W_OUT
    tl.max_contiguous(offs_w, BLOCK_W)

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    scale1_val = tl.load(scale1_ptr).to(tl.float32)
    bias_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0).to(tl.float32)

    t_d_base = od + PAD_D
    t_h_base = oh + PAD_H

    for ic in tl.range(0, C_IN):
        for kd in tl.static_range(0, 3):
            t_d = t_d_base - kd
            even_d = (t_d & 1) == 0
            id_ = t_d // STRIDE_D
            valid_d = even_d & (id_ >= 0) & (id_ < D_IN)

            for kh in tl.static_range(0, 3):
                t_h = t_h_base - kh
                even_h = (t_h & 1) == 0
                ih = t_h // STRIDE_H
                valid_h = even_h & (ih >= 0) & (ih < H_IN)

                x_base = (
                    x_ptr
                    + n * x_stride_n
                    + ic * x_stride_c
                    + id_ * x_stride_d
                    + ih * x_stride_h
                )
                w_base = (
                    w_ptr
                    + ic * w_stride_ci
                    + kd * w_stride_kd
                    + kh * w_stride_kh
                )

                for kw in tl.static_range(0, 3):
                    t_w = offs_w + PAD_W - kw
                    even_w = (t_w & 1) == 0
                    iw = t_w // STRIDE_W
                    valid_w = even_w & (iw >= 0) & (iw < W_IN) & w_mask

                    x_ptrs = x_base + iw * x_stride_w
                    x_vals = tl.load(
                        x_ptrs,
                        mask=valid_d & valid_h & valid_w,
                        other=0.0,
                    ).to(tl.float32)

                    w_ptrs = w_base + kw * w_stride_kw + offs_oc * w_stride_co
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0).to(tl.float32)

                    acc += w_vals[:, None] * x_vals[None, :]

    acc = (acc + bias_vals[:, None]) * scale1_val

    y_base = (
        y_ptr
        + n * y_stride_n
        + od * y_stride_d
        + oh * y_stride_h
    )
    y_ptrs = y_base + offs_oc[:, None] * y_stride_c + offs_w[None, :] * y_stride_w
    store_mask = oc_mask[:, None] & w_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


@triton.jit
def _avgpool3d_add_mul2_kernel(
    x_ptr, bias2_ptr, scale2_ptr, y_ptr,
    N, C, D, H, W,
    D_OUT, H_OUT, W_OUT,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    bias2_stride_c,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_rest = tl.program_id(1)

    oh = pid_rest % H_OUT
    tmp = pid_rest // H_OUT
    od = tmp % D_OUT
    tmp = tmp // D_OUT
    c_blk = tmp % tl.cdiv(C, BLOCK_C)
    n = tmp // tl.cdiv(C, BLOCK_C)

    offs_c = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_OUT
    tl.max_contiguous(offs_w, BLOCK_W)

    in_d0 = od * 2
    in_h0 = oh * 2
    in_w0 = offs_w * 2

    x_base = (
        x_ptr
        + n * x_stride_n
        + offs_c[:, None] * x_stride_c
        + in_d0 * x_stride_d
        + in_h0 * x_stride_h
        + in_w0[None, :] * x_stride_w
    )

    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            ptr0 = x_base + dd * x_stride_d + hh * x_stride_h
            v0 = tl.load(ptr0 + 0 * x_stride_w, mask=mask_c[:, None] & mask_w[None, :], other=0.0).to(tl.float32)
            v1 = tl.load(ptr0 + 1 * x_stride_w, mask=mask_c[:, None] & mask_w[None, :], other=0.0).to(tl.float32)
            acc += v0 + v1
    acc *= 0.125

    b = tl.load(bias2_ptr + offs_c * bias2_stride_c, mask=mask_c, other=0.0).to(tl.float32)
    scale2_val = tl.load(scale2_ptr).to(tl.float32)
    out = (acc + b[:, None]) * scale2_val

    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + offs_c[:, None] * y_stride_c
        + od * y_stride_d
        + oh * y_stride_h
        + offs_w[None, :] * y_stride_w
    )
    tl.store(y_ptrs, out, mask=mask_c[:, None] & mask_w[None, :])


def kernel_function(x, conv_weight, conv_bias, scale1, bias2, scale2):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU not available.")

    x_xpu = x if x.device.type == "xpu" else x.to("xpu", dtype=torch.float16)
    conv_weight_xpu = conv_weight if conv_weight.device.type == "xpu" else conv_weight.to("xpu", dtype=torch.float16)
    conv_bias_xpu = conv_bias if conv_bias.device.type == "xpu" else conv_bias.to("xpu", dtype=torch.float16)
    scale1_xpu = scale1 if scale1.device.type == "xpu" else scale1.to("xpu", dtype=torch.float16)
    bias2_xpu = bias2 if bias2.device.type == "xpu" else bias2.to("xpu", dtype=torch.float16)
    scale2_xpu = scale2 if scale2.device.type == "xpu" else scale2.to("xpu", dtype=torch.float16)

    if x_xpu.dtype != torch.float16:
        x_xpu = x_xpu.to(torch.float16)
    if conv_weight_xpu.dtype != torch.float16:
        conv_weight_xpu = conv_weight_xpu.to(torch.float16)
    if conv_bias_xpu.dtype != torch.float16:
        conv_bias_xpu = conv_bias_xpu.to(torch.float16)
    if scale1_xpu.dtype != torch.float16:
        scale1_xpu = scale1_xpu.to(torch.float16)
    if bias2_xpu.dtype != torch.float16:
        bias2_xpu = bias2_xpu.to(torch.float16)
    if scale2_xpu.dtype != torch.float16:
        scale2_xpu = scale2_xpu.to(torch.float16)

    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()
    if not conv_weight_xpu.is_contiguous():
        conv_weight_xpu = conv_weight_xpu.contiguous()
    if not conv_bias_xpu.is_contiguous():
        conv_bias_xpu = conv_bias_xpu.contiguous()
    if not scale1_xpu.is_contiguous():
        scale1_xpu = scale1_xpu.contiguous()
    if not bias2_xpu.is_contiguous():
        bias2_xpu = bias2_xpu.contiguous()
    if not scale2_xpu.is_contiguous():
        scale2_xpu = scale2_xpu.contiguous()

    N, C_IN, D_IN, H_IN, W_IN = x_xpu.shape
    w_cin, C_OUT, KD, KH, KW = conv_weight_xpu.shape
    assert w_cin == C_IN

    stride_d, stride_h, stride_w = 2, 2, 2
    pad_d, pad_h, pad_w = 1, 1, 1
    out_pad = (0, 0, 0)
    dil = (1, 1, 1)

    D_OUT1 = (D_IN - 1) * stride_d - 2 * pad_d + dil[0] * (KD - 1) + out_pad[0] + 1
    H_OUT1 = (H_IN - 1) * stride_h - 2 * pad_h + dil[1] * (KH - 1) + out_pad[1] + 1
    W_OUT1 = (W_IN - 1) * stride_w - 2 * pad_w + dil[2] * (KW - 1) + out_pad[2] + 1

    y1 = torch.empty((N, C_OUT, D_OUT1, H_OUT1, W_OUT1), device="xpu", dtype=torch.float16)

    BLOCK_W0 = 64
    BLOCK_OC0 = 16
    NUM_WARPS0 = 8
    grid_conv = (
        triton.cdiv(W_OUT1, BLOCK_W0),
        N * D_OUT1 * H_OUT1,
        triton.cdiv(C_OUT, BLOCK_OC0),
    )
    _conv_transpose3d_mul1_kernel[grid_conv](
        x_xpu, conv_weight_xpu, conv_bias_xpu, scale1_xpu, y1,
        N, C_IN, D_IN, H_IN, W_IN,
        C_OUT, KD, KH, KW,
        D_OUT1, H_OUT1, W_OUT1,
        STRIDE_D=stride_d, STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_D=pad_d, PAD_H=pad_h, PAD_W=pad_w,
        x_stride_n=x_xpu.stride(0), x_stride_c=x_xpu.stride(1),
        x_stride_d=x_xpu.stride(2), x_stride_h=x_xpu.stride(3), x_stride_w=x_xpu.stride(4),
        w_stride_ci=conv_weight_xpu.stride(0), w_stride_co=conv_weight_xpu.stride(1),
        w_stride_kd=conv_weight_xpu.stride(2), w_stride_kh=conv_weight_xpu.stride(3),
        w_stride_kw=conv_weight_xpu.stride(4),
        y_stride_n=y1.stride(0), y_stride_c=y1.stride(1),
        y_stride_d=y1.stride(2), y_stride_h=y1.stride(3), y_stride_w=y1.stride(4),
        BLOCK_W=BLOCK_W0, BLOCK_OC=BLOCK_OC0, NUM_WARPS=NUM_WARPS0,
        grf_mode="auto",
        num_warps=NUM_WARPS0, num_stages=2,
    )

    N1, C1, D1, H1, W1 = y1.shape
    D_OUT2 = (D1 - 2) // 2 + 1
    H_OUT2 = (H1 - 2) // 2 + 1
    W_OUT2 = (W1 - 2) // 2 + 1
    y2 = torch.empty((N1, C1, D_OUT2, H_OUT2, W_OUT2), device="xpu", dtype=torch.float16)

    BLOCK_W1 = 64
    BLOCK_C1 = 8
    grid_pool = (
        triton.cdiv(W_OUT2, BLOCK_W1),
        N1 * triton.cdiv(C1, BLOCK_C1) * D_OUT2 * H_OUT2,
    )
    _avgpool3d_add_mul2_kernel[grid_pool](
        y1, bias2_xpu, scale2_xpu, y2,
        N1, C1, D1, H1, W1,
        D_OUT2, H_OUT2, W_OUT2,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3), y1.stride(4),
        bias2_xpu.stride(0),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3), y2.stride(4),
        BLOCK_W=BLOCK_W1,
        BLOCK_C=BLOCK_C1,
        grf_mode="auto",
        num_warps=8, num_stages=2,
    )
    return y2


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(float(scale1), dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float16))
        self.scale2 = nn.Parameter(torch.tensor(float(scale2), dtype=torch.float16))
        self.stride = stride
        self.padding = padding
        self._moved_to_xpu = False

    def _move_to_xpu_once(self):
        if self._moved_to_xpu:
            return
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
        if self.conv_transpose.bias is not None:
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float16).contiguous()
        self.scale1.data = self.scale1.data.to("xpu", dtype=torch.float16).contiguous()
        self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()
        self.scale2.data = self.scale2.data.to("xpu", dtype=torch.float16).contiguous()
        self._moved_to_xpu = True

    def forward(self, x):
        self._move_to_xpu_once()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()
        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.scale1,
            self.bias,
            self.scale2,
        )