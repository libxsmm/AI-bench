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
def _conv_transpose3d_kernel(
    x,
    weight,
    output,
    B: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    D_IN: ConstInt,
    H_IN: ConstInt,
    W_IN: ConstInt,
    D_OUT: ConstInt,
    H_OUT: ConstInt,
    W_OUT: ConstInt,
    KD: ConstInt,
    KH: ConstInt,
    KW: ConstInt,
    STRIDE_D: ConstInt,
    STRIDE_H: ConstInt,
    STRIDE_W: ConstInt,
    PAD_D: ConstInt,
    PAD_H: ConstInt,
    PAD_W: ConstInt,
    GROUPS: ConstInt,
    C_IN_PG: ConstInt,
    C_OUT_PG: ConstInt,
    BLOCK_SZ: ConstInt,
):
    batch = ct.bid(0)
    output_channel = ct.bid(1)
    spatial_block = ct.bid(2)
    lanes = spatial_block * BLOCK_SZ + ct.arange(BLOCK_SZ, dtype=torch.int32)
    spatial_valid = lanes < D_OUT * H_OUT * W_OUT
    depth = lanes // (H_OUT * W_OUT)
    rem = lanes % (H_OUT * W_OUT)
    height = rem // W_OUT
    width = rem % W_OUT
    group = output_channel // C_OUT_PG
    output_local = output_channel % C_OUT_PG
    input_channel_start = group * C_IN_PG
    acc = ct.full((BLOCK_SZ,), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    max_x_offset = B * C_IN * D_IN * H_IN * W_IN - 1
    max_weight_offset = C_IN * C_OUT_PG * KD * KH * KW - 1

    for input_local in range(C_IN_PG):
        input_channel = input_channel_start + input_local
        for kd in range(KD):
            depth_num = depth + PAD_D - kd
            input_depth = depth_num // STRIDE_D
            valid_depth = (depth_num == input_depth * STRIDE_D) & (input_depth >= 0) & (input_depth < D_IN)
            for kh in range(KH):
                height_num = height + PAD_H - kh
                input_height = height_num // STRIDE_H
                valid_height = (height_num == input_height * STRIDE_H) & (input_height >= 0) & (input_height < H_IN)
                for kw in range(KW):
                    width_num = width + PAD_W - kw
                    input_width = width_num // STRIDE_W
                    valid_width = (width_num == input_width * STRIDE_W) & (input_width >= 0) & (input_width < W_IN)
                    valid = spatial_valid & valid_depth & valid_height & valid_width
                    x_indices = ((((batch * C_IN + input_channel) * D_IN + input_depth) * H_IN + input_height) * W_IN + input_width)
                    safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                    x_value = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0).astype(ct.float32)
                    weight_index = ((((input_channel * C_OUT_PG + output_local) * KD + kd) * KH + kh) * KW + kw)
                    safe_weight_index = ct.minimum(ct.maximum(weight_index, 0), max_weight_offset)
                    weight_value = weight_mem.load_offset(safe_weight_index + ct.arange(1, dtype=torch.int32), mask=safe_weight_index >= 0).astype(ct.float32)
                    acc += x_value * weight_value * valid.astype(ct.float32)

    output_index = ((((batch * C_OUT + output_channel) * D_OUT + depth) * H_OUT + height) * W_OUT + width)
    safe_output_index = ct.minimum(ct.maximum(output_index, 0), B * C_OUT * D_OUT * H_OUT * W_OUT - 1)
    output.get_raw_memory().store_offset(safe_output_index, ct.astype(acc, output.dtype), mask=spatial_valid)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False, dilation=(1,1,1)):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous()
        B, C_IN, D_IN, H_IN, W_IN = x.shape
        KD, KH, KW = self.conv.kernel_size
        STRIDE_D, STRIDE_H, STRIDE_W = self.conv.stride
        PAD_D, PAD_H, PAD_W = self.conv.padding
        D_OUT = (D_IN - 1) * STRIDE_D - 2 * PAD_D + KD + self.conv.output_padding[0]
        H_OUT = (H_IN - 1) * STRIDE_H - 2 * PAD_H + KH + self.conv.output_padding[1]
        W_OUT = (W_IN - 1) * STRIDE_W - 2 * PAD_W + KW + self.conv.output_padding[2]
        C_OUT = self.conv.out_channels
        C_IN_PG = C_IN // self.conv.groups
        C_OUT_PG = C_OUT // self.conv.groups
        weight = self.conv.weight.to(dtype=torch.float16).contiguous()
        output = torch.empty((B, C_OUT, D_OUT, H_OUT, W_OUT), device=x.device, dtype=torch.float16)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (B, C_OUT, ct.cdiv(D_OUT * H_OUT * W_OUT, 32)), _conv_transpose3d_kernel, (x, weight, output, B, C_IN, C_OUT, D_IN, H_IN, W_IN, D_OUT, H_OUT, W_OUT, KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, self.conv.groups, C_IN_PG, C_OUT_PG, 32))
        return output
