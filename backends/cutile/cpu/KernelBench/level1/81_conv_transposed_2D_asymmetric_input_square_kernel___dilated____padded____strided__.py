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
def _conv_transpose2d_scatter_kernel(x, weight, output, B: ConstInt, H_IN: ConstInt, W_IN: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, H_OUT: ConstInt, W_OUT: ConstInt, KH: ConstInt, KW: ConstInt, STRIDE_H: ConstInt, STRIDE_W: ConstInt, PAD_H: ConstInt, PAD_W: ConstInt, DIL_H: ConstInt, DIL_W: ConstInt, BLOCK_W: ConstInt, BLOCK_K: ConstInt, BLOCK_N: ConstInt):
    pid_w = ct.bid(0); pid_bh = ct.bid(1); pid_n = ct.bid(2)
    batch = pid_bh // H_IN; ih = pid_bh % H_IN
    rows = pid_w * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32); cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < W_IN; col_valid = cols < C_OUT
    x_mem = x.get_raw_memory(); weight_mem = weight.get_raw_memory(); output_mem = output.get_raw_memory()
    cin = ct.arange(BLOCK_K, dtype=torch.int32); valid_c = cin < C_IN
    x_indices = (((batch * H_IN + ih) * W_IN + rows[:, None]) * C_IN) + cin[None, :]
    safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), B * H_IN * W_IN * C_IN - 1)
    x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0) * (row_valid[:, None] & valid_c[None, :]).astype(torch.float16)
    max_weight_offset = KH * KW * C_IN * C_OUT - 1
    max_output_offset = B * H_OUT * W_OUT * C_OUT - 1
    for kh in range(KH):
        output_h = ih * STRIDE_H + kh * DIL_H - PAD_H
        valid_h = (output_h >= 0) & (output_h < H_OUT)
        for kw in range(KW):
            output_w = rows * STRIDE_W + kw * DIL_W - PAD_W
            valid_output = (row_valid & valid_h & (output_w >= 0) & (output_w < W_OUT))[:, None] & col_valid[None, :]
            kernel_index = kh * KW + kw
            weight_base = kernel_index * C_IN * C_OUT + cin[:, None] * C_OUT + cols[None, :]
            safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), max_weight_offset)
            weight_values = weight_mem.load_offset(safe_weight_base, mask=valid_c[:, None] & col_valid[None, :], padding_value=0.0)
            result = ct.mma(x_values, weight_values, ct.full((BLOCK_W, BLOCK_N), 0.0, dtype=ct.float32))
            output_indices = (((batch * H_OUT + output_h) * W_OUT + output_w[:, None]) * C_OUT) + cols[None, :]
            safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), max_output_offset)
            output_mem.store_offset(safe_output_indices, ct.astype(result, output.dtype), mask=valid_output)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous(); B, C_IN, H_IN, W_IN = x.shape
        KH, KW = self.conv.kernel_size; SH, SW = self.conv.stride; PH, PW = self.conv.padding; DH, DW = self.conv.dilation
        H_OUT = (H_IN - 1) * SH - 2 * PH + DH * (KH - 1) + 1 + self.conv.output_padding[0]; W_OUT = (W_IN - 1) * SW - 2 * PW + DW * (KW - 1) + 1 + self.conv.output_padding[1]
        x_channels_last = x.contiguous(memory_format=torch.channels_last)
        weight = self.conv.weight.permute(2, 3, 0, 1).contiguous().to(dtype=torch.float16)
        output = torch.zeros((B, self.conv.out_channels, H_OUT, W_OUT), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(W_IN, 16), B * H_IN, ct.cdiv(self.conv.out_channels, 32)), _conv_transpose2d_scatter_kernel, (x_channels_last, weight, output, B, H_IN, W_IN, C_IN, self.conv.out_channels, H_OUT, W_OUT, KH, KW, SH, SW, PH, PW, DH, DW, 16, 32, 32))
        return output
