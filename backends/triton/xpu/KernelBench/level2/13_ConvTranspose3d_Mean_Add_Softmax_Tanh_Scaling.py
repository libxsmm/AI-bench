# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_in_b: tl.constexpr,
    stride_in_c: tl.constexpr,
    stride_in_d: tl.constexpr,
    stride_in_h: tl.constexpr,
    stride_in_w: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_c: tl.constexpr,
    stride_out_d: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_w: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    HW = H * W
    b = pid0 // HW
    rem = pid0 % HW
    h = rem // W
    w = rem % W
    c = pid1

    if b >= B or c >= C or h >= H or w >= W:
        return

    off_in = b * stride_in_b + c * stride_in_c + h * stride_in_h + w * stride_in_w

    sum_val = 0.0
    for d in range(0, D):
        val = tl.load(input_ptr + off_in + d * stride_in_d)
        sum_val += val

    mean = sum_val / D

    off_out = b * stride_out_b + c * stride_out_c + h * stride_out_h + w * stride_out_w
    tl.store(output_ptr + off_out, mean)


@triton.jit
def add_bias_kernel(
    mean_ptr,
    bias_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_m_b: tl.constexpr,
    stride_m_c: tl.constexpr,
    stride_m_h: tl.constexpr,
    stride_m_w: tl.constexpr,
    stride_b_c: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_c: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_w: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    HW = H * W
    b = pid0 // HW
    rem = pid0 % HW
    h = rem // W
    w = rem % W
    c = pid1

    if b >= B or c >= C or h >= H or w >= W:
        return

    off_m = b * stride_m_b + c * stride_m_c + h * stride_m_h + w * stride_m_w
    val = tl.load(mean_ptr + off_m)

    bval = tl.load(bias_ptr + c * stride_b_c)
    res = val + bval

    off_o = b * stride_o_b + c * stride_o_c + h * stride_o_h + w * stride_o_w
    tl.store(output_ptr + off_o, res)


@triton.jit
def softmax_tanh_mul_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    scale,
    stride_i_b: tl.constexpr,
    stride_i_c: tl.constexpr,
    stride_i_h: tl.constexpr,
    stride_i_w: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_c: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_w: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    HW = H * W
    b = pid0 // HW
    rem = pid0 % HW
    h = rem // W
    w = rem % W
    c = pid1

    if b >= B or c >= C or h >= H or w >= W:
        return

    base = b * stride_i_b + h * stride_i_h + w * stride_i_w

    max_val = -float("inf")
    for cc in range(0, C):
        v = tl.load(input_ptr + base + cc * stride_i_c)
        max_val = tl.maximum(max_val, v)

    sum_exp = 0.0
    for cc in range(0, C):
        v = tl.load(input_ptr + base + cc * stride_i_c)
        sum_exp += tl.exp(v - max_val)

    v_cur = tl.load(input_ptr + base + c * stride_i_c)
    y = tl.exp(v_cur - max_val) / sum_exp

    y2 = y * y
    tanh_y = y * (27.0 + y2) / (27.0 + 9.0 * y2)

    y = tanh_y * scale

    off_o = b * stride_o_b + c * stride_o_c + h * stride_o_h + w * stride_o_w
    tl.store(output_ptr + off_o, y)


@triton.jit
def fused_epilogue_split_softmax_kernel(
    input_ptr,
    bias_ptr,
    row_max_ptr,
    row_sum_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_in_b,
    stride_in_c,
    stride_in_d,
    stride_in_h,
    stride_in_w,
    stride_bias_c,
    stride_out_b,
    stride_out_c,
    stride_out_d,
    stride_out_h,
    stride_out_w,
    scale,
    BLOCK_C: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)

    HW = H * W
    b = pid // HW
    rem = pid - b * HW
    h = rem // W
    w = rem - h * W

    if b >= B:
        return

    base_in = input_ptr + b * stride_in_b + h * stride_in_h + w * stride_in_w
    base_out = output_ptr + b * stride_out_b + h * stride_out_h + w * stride_out_w

    in_bp = tl.make_block_ptr(
        base=base_in,
        shape=(C, D),
        strides=(stride_in_c, stride_in_d),
        offsets=(0, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )

    x_block = tl.load(in_bp, boundary_check=(0, 1))
    acc = tl.sum(x_block.to(tl.float32), axis=1)
    mean_vals = acc * (1.0 / D)

    bias_bp = tl.make_block_ptr(
        base=bias_ptr,
        shape=(C, 1),
        strides=(stride_bias_c, 0),
        offsets=(0, 0),
        block_shape=(BLOCK_C, 1),
        order=(1, 0),
    )
    bias_vals = tl.load(bias_bp, boundary_check=(0, 1)).to(tl.float32)
    bias_vals = tl.reshape(bias_vals, (BLOCK_C,))

    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    logits = mean_vals + bias_vals
    logits_masked = tl.where(c_mask, logits, float("-inf"))
    row_max = tl.max(logits_masked, axis=0)
    tl.store(row_max_ptr + pid, row_max)

    exp_vals = tl.exp(logits - row_max)
    exp_vals = tl.where(c_mask, exp_vals, 0.0)
    row_sum = tl.sum(exp_vals, axis=0)
    tl.store(row_sum_ptr + pid, row_sum)

    y = exp_vals / row_sum
    y2 = y * y
    tanh_y = y * (27.0 + y2) / (27.0 + 9.0 * y2)
    out_vals = tanh_y * scale
    out_vals = out_vals.to(output_ptr.type.element_ty)

    out_bp = tl.make_block_ptr(
        base=base_out,
        shape=(C, 1),
        strides=(stride_out_c, stride_out_d),
        offsets=(0, 0),
        block_shape=(BLOCK_C, 1),
        order=(1, 0),
    )
    tl.store(out_bp, tl.reshape(out_vals, (BLOCK_C, 1)), boundary_check=(0, 1))


def kernel_function(x, bias, scaling_factor):
    assert x.ndim == 5 and bias.ndim == 5

    if x.device.type != "xpu":
        x_xpu = x.to("xpu", dtype=torch.float16)
    elif x.dtype != torch.float16:
        x_xpu = x.to(dtype=torch.float16)
    else:
        x_xpu = x
    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()

    if bias.device.type != "xpu":
        bias_xpu = bias.to("xpu", dtype=torch.float16)
    elif bias.dtype != torch.float16:
        bias_xpu = bias.to(dtype=torch.float16)
    else:
        bias_xpu = bias
    if not bias_xpu.is_contiguous():
        bias_xpu = bias_xpu.contiguous()

    B, C, D, H, W = x_xpu.shape
    out = torch.empty((B, C, 1, H, W), device=x_xpu.device, dtype=x_xpu.dtype)

    sx = x_xpu.stride()
    sb = bias_xpu.stride()
    so = out.stride()

    rows = B * H * W
    row_max = torch.empty((rows,), device=x_xpu.device, dtype=torch.float32)
    row_sum = torch.empty((rows,), device=x_xpu.device, dtype=torch.float32)

    BLOCK_C = 64
    grid = (rows,)

    fused_epilogue_split_softmax_kernel[grid](
        x_xpu,
        bias_xpu,
        row_max,
        row_sum,
        out,
        B,
        C,
        D,
        H,
        W,
        sx[0],
        sx[1],
        sx[2],
        sx[3],
        sx[4],
        sb[1],
        so[0],
        so[1],
        so[2],
        so[3],
        so[4],
        scaling_factor,
        BLOCK_C=BLOCK_C,
        grf_mode="auto",
        num_warps=4,
        num_stages=1,
    )

    return out


batch_size = 16
in_channels = 16
out_channels = 64
depth = 32
height = width = 128
kernel_size = 3
stride = 1
padding = 1
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        if x.device.type != "xpu":
            x = x.to("xpu", dtype=torch.float16)
        elif x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)

        if self.conv_transpose.weight.device.type != "xpu" or self.conv_transpose.weight.dtype != torch.float16:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
        if self.conv_transpose.bias is not None and (
            self.conv_transpose.bias.device.type != "xpu" or self.conv_transpose.bias.dtype != torch.float16
        ):
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float16).contiguous()
        if self.bias.device.type != "xpu" or self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()

        y = self.conv_transpose(x)
        return kernel_function(y, self.bias, self.scaling_factor)