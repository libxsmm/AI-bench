# ruff: noqa: E731
# Example CUDA Tile CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import torch
import torch.nn as nn
import cuda.tile as ct
from cuda.tile._backend import cpu

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv_transpose2d_kernel(
    x,
    weight,
    bias,
    output,
    TOTAL: ConstInt,
    B: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    H_IN: ConstInt,
    W_IN: ConstInt,
    H_OUT: ConstInt,
    W_OUT: ConstInt,
    KH: ConstInt,
    KW: ConstInt,
    STRIDE_H: ConstInt,
    STRIDE_W: ConstInt,
    PAD_H: ConstInt,
    PAD_W: ConstInt,
    DILATION_H: ConstInt,
    DILATION_W: ConstInt,
    GROUPS: ConstInt,
    C_IN_PG: ConstInt,
    C_OUT_PG: ConstInt,
    BLOCK: ConstInt,
):
    pid = ct.bid(0)
    lanes = pid * BLOCK + ct.arange(BLOCK, dtype=torch.int32)
    output_valid = lanes < TOTAL
    nc = C_OUT * H_OUT * W_OUT
    batch = lanes // nc
    rem = lanes % nc
    output_channel = rem // (H_OUT * W_OUT)
    rem = rem % (H_OUT * W_OUT)
    output_height = rem // W_OUT
    output_width = rem % W_OUT
    group = output_channel // C_OUT_PG
    output_local = output_channel % C_OUT_PG
    input_start = group * C_IN_PG
    acc = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    max_x_offset = B * C_IN * H_IN * W_IN - 1
    max_weight_offset = C_IN * C_OUT_PG * KH * KW - 1
    for kh in range(KH):
        input_height_num = output_height + PAD_H - kh * DILATION_H
        input_height = input_height_num // STRIDE_H
        valid_height = (input_height_num == input_height * STRIDE_H) & (input_height >= 0) & (input_height < H_IN)
        for kw in range(KW):
            input_width_num = output_width + PAD_W - kw * DILATION_W
            input_width = input_width_num // STRIDE_W
            valid = output_valid & valid_height & (input_width_num == input_width * STRIDE_W) & (input_width >= 0) & (input_width < W_IN)
            for ci in range(C_IN_PG):
                input_channel = input_start + ci
                x_indices = (((batch * C_IN + input_channel) * H_IN + input_height) * W_IN + input_width)
                safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                x_value = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0).astype(ct.float32)
                weight_index = (((input_channel * C_OUT_PG + output_local) * KH + kh) * KW + kw)
                safe_weight_index = ct.minimum(ct.maximum(weight_index, 0), max_weight_offset)
                weight_value = weight_mem.load_offset(safe_weight_index, mask=safe_weight_index >= 0).astype(ct.float32)
                acc += x_value * weight_value * valid.astype(ct.float32)
    safe_bias_index = ct.minimum(ct.maximum(output_channel, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_bias_index, mask=output_valid, padding_value=0.0).astype(ct.float32)
    output_index = (((batch * C_OUT + output_channel) * H_OUT + output_height) * W_OUT + output_width)
    safe_output_index = ct.minimum(ct.maximum(output_index, 0), TOTAL - 1)
    output.get_raw_memory().store_offset(safe_output_index, ct.astype(acc, output.dtype), mask=output_valid)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=0, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous()
        B, C_IN, H_IN, W_IN = x.shape
        KH, KW = self.conv.kernel_size
        STRIDE_H, STRIDE_W = self.conv.stride
        PAD_H, PAD_W = self.conv.padding
        DILATION_H, DILATION_W = self.conv.dilation
        H_OUT = (H_IN - 1) * STRIDE_H - 2 * PAD_H + DILATION_H * (KH - 1) + 1
        W_OUT = (W_IN - 1) * STRIDE_W - 2 * PAD_W + DILATION_W * (KW - 1) + 1
        C_OUT = self.conv.out_channels
        C_IN_PG = C_IN // self.conv.groups
        C_OUT_PG = C_OUT // self.conv.groups
        weight = self.conv.weight.to(dtype=torch.float16).contiguous()
        bias = torch.zeros(C_OUT, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.empty((B, C_OUT, H_OUT, W_OUT), device=x.device, dtype=torch.float16)
        total = B * C_OUT * H_OUT * W_OUT
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(total, 32),), _conv_transpose2d_kernel, (x, weight, bias, output, total, B, C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, DILATION_H, DILATION_W, self.conv.groups, C_IN_PG, C_OUT_PG, 32))
        return output
