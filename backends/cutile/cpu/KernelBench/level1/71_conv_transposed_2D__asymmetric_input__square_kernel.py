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
def _conv_transpose2d_kernel(x, weight, bias, output, B: ConstInt, H_IN: ConstInt, W_IN: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, H_OUT: ConstInt, W_OUT: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_W: ConstInt, BLOCK_K: ConstInt, BLOCK_N: ConstInt):
    pid = ct.bid(0); batch = ct.bid(1); num_w = ct.cdiv(W_OUT, BLOCK_W); num_n = ct.cdiv(C_OUT, BLOCK_N)
    m = pid // num_n; pid_n = pid % num_n; oh = m // num_w; pid_w = m % num_w
    rows = pid_w * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32); cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < W_OUT; col_valid = cols < C_OUT
    acc = ct.full((BLOCK_W, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory(); weight_mem = weight.get_raw_memory(); bias_mem = bias.get_raw_memory()
    cin = ct.arange(BLOCK_K, dtype=torch.int32); valid_c = cin < C_IN
    max_x_offset = B * H_IN * W_IN * C_IN - 1; max_weight_offset = KH * KW * C_IN * C_OUT - 1
    for kh in range(KH):
        input_h = oh - kh
        for kw in range(KW):
            input_w = rows - kw
            valid = row_valid & (input_h >= 0) & (input_h < H_IN) & (input_w >= 0) & (input_w < W_IN)
            x_indices = (((batch * H_IN + input_h) * W_IN + input_w[:, None]) * C_IN) + cin[None, :]
            safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
            x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0) * (valid[:, None] & valid_c[None, :]).astype(torch.float16)
            kernel_index = kh * KW + kw; weight_base = kernel_index * C_IN * C_OUT + cin[:, None] * C_OUT + cols[None, :]
            safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), max_weight_offset)
            weight_values = weight_mem.load_offset(safe_weight_base, mask=valid_c[:, None] & col_valid[None, :], padding_value=0.0)
            acc = ct.mma(x_values, weight_values, acc)
    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1); acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = (((batch * H_OUT + oh) * W_OUT + rows[:, None]) * C_OUT) + cols[None, :]
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), B * H_OUT * W_OUT * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output_indices, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, (kernel_size,kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous(); B, C_IN, H_IN, W_IN = x.shape
        KH, KW = self.conv.kernel_size; H_OUT = H_IN + KH - 1; W_OUT = W_IN + KW - 1
        x_channels_last = x.contiguous(memory_format=torch.channels_last)
        weight = self.conv.weight.permute(2, 3, 0, 1).contiguous().to(dtype=torch.float16)
        bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.empty((B, self.conv.out_channels, H_OUT, W_OUT), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (H_OUT * ct.cdiv(W_OUT, 64) * ct.cdiv(self.conv.out_channels, 32), B), _conv_transpose2d_kernel, (x_channels_last, weight, bias, output, B, H_IN, W_IN, C_IN, self.conv.out_channels, H_OUT, W_OUT, KH, KW, 64, 32, 32))
        return output
