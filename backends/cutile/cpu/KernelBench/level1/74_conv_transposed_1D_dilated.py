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
def _conv_transpose1d_dilated_kernel(x, weight, bias, output, B: ConstInt, L_IN: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, L_OUT: ConstInt, K_SIZE: ConstInt, DILATION: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    pid = ct.bid(0)
    num_n = ct.cdiv(C_OUT, BLOCK_N)
    rows = (pid // num_n) * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = (pid % num_n) * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < B * L_OUT
    col_valid = cols < C_OUT
    batch = rows // L_OUT
    ol = rows % L_OUT
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    cin = ct.arange(C_IN, dtype=torch.int32)
    max_x_offset = B * L_IN * C_IN - 1
    max_weight_offset = K_SIZE * C_IN * C_OUT - 1
    for k in range(K_SIZE):
        il = ol - k * DILATION
        valid = row_valid & (il >= 0) & (il < L_IN)
        x_indices = ((batch[:, None] * L_IN + il[:, None]) * C_IN) + cin[None, :]
        safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
        x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0)
        x_values = x_values * valid[:, None].astype(torch.float16)
        weight_base = k * C_IN * C_OUT + cin[:, None] * C_OUT + cols[None, :]
        safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), max_weight_offset)
        weight_values = weight_mem.load_offset(safe_weight_base, mask=col_valid[None, :], padding_value=0.0)
        acc = ct.mma(x_values, weight_values, acc)
    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = rows[:, None] * C_OUT + cols[None, :]
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), B * L_OUT * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output_indices, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous()
        B, C_IN, L_IN = x.shape
        K_SIZE = self.conv.kernel_size[0]
        DILATION = self.conv.dilation[0]
        L_OUT = L_IN + DILATION * (K_SIZE - 1)
        weight = self.conv.weight.permute(2, 0, 1).contiguous().to(dtype=torch.float16)
        bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.empty((B, L_OUT, self.conv.out_channels), device=x.device, dtype=torch.float16)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(B * L_OUT, 32) * ct.cdiv(self.conv.out_channels, 32),), _conv_transpose1d_dilated_kernel, (x.permute(0, 2, 1).contiguous(), weight, bias, output, B, L_IN, C_IN, self.conv.out_channels, L_OUT, K_SIZE, DILATION, 32, 32))
        return output.permute(0, 2, 1).contiguous()
