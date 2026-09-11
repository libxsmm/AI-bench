# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv1d_kernel(x, weight, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, L_IN: ConstInt, L_OUT: ConstInt, K_SIZE: ConstInt, BLOCK_OL: ConstInt, BLOCK_N: ConstInt):
    batch = ct.bid(0)
    pid_ol = ct.bid(1)
    pid_n = ct.bid(2)
    positions = pid_ol * BLOCK_OL + ct.arange(BLOCK_OL, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    position_valid = positions < L_OUT
    col_valid = cols < C_OUT
    acc = ct.full((BLOCK_OL, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    max_x = B * L_IN * C_IN - 1
    max_weight = K_SIZE * C_IN * C_OUT - 1
    for k in range(K_SIZE):
        for block in range(ct.cdiv(C_IN, 16)):
            cin = block * 16 + ct.arange(16, dtype=torch.int32)
            cin_valid = cin < C_IN
            x_indices = ((batch * L_IN + positions[:, None] + k) * C_IN + cin[None, :])
            safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x)
            x_tile = x_mem.load_offset(safe_x, mask=safe_x >= 0)
            x_tile = x_tile * (position_valid[:, None] & cin_valid[None, :]).astype(torch.float16)
            weight_indices = (k * C_IN + cin[:, None]) * C_OUT + cols[None, :]
            safe_weight = ct.minimum(ct.maximum(weight_indices, 0), max_weight)
            weight_tile = weight_mem.load_offset(safe_weight, mask=cin_valid[:, None] & col_valid[None, :], padding_value=0.0)
            acc = ct.mma(x_tile, weight_tile, acc)
    output_indices = ((batch * L_OUT + positions[:, None]) * C_OUT + cols[None, :])
    safe_output = ct.minimum(ct.maximum(output_indices, 0), B * L_OUT * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output, ct.astype(acc, output.dtype), mask=position_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.conv1d.stride != (1,) or self.conv1d.padding != (0,) or self.conv1d.dilation != (1,) or self.conv1d.groups != 1 or self.conv1d.bias is not None:
            return torch.nn.functional.conv1d(x.to(torch.float16), self.conv1d.weight.to(torch.float16), None if self.conv1d.bias is None else self.conv1d.bias.to(torch.float16), self.conv1d.stride, self.conv1d.padding, self.conv1d.dilation, self.conv1d.groups)
        x = x.to(torch.float16)
        B, C_IN, L_IN = x.shape
        K_SIZE = self.conv1d.kernel_size[0]
        L_OUT = L_IN - K_SIZE + 1
        weight = self.conv1d.weight.permute(2, 1, 0).contiguous().to(torch.float16)
        output = torch.empty((B, L_OUT, self.conv1d.out_channels), device=x.device, dtype=torch.float16)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (B, ct.cdiv(L_OUT, 64), ct.cdiv(self.conv1d.out_channels, 128)), _conv1d_kernel, (x.permute(0, 2, 1).contiguous(), weight, output, B, C_IN, self.conv1d.out_channels, L_IN, L_OUT, K_SIZE, 64, 128))
        return output.permute(0, 2, 1).contiguous()
