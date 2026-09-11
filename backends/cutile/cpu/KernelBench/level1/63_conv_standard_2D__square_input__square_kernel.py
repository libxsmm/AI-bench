# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv2d_flat_kernel(x, weight, bias, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, H: ConstInt, W: ConstInt, OH: ConstInt, OW: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    rows = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < B * OH * OW
    col_valid = cols < C_OUT
    batch = rows // (OH * OW)
    rem = rows % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    max_x = B * H * W * C_IN - 1
    max_w = KH * KW * C_IN * C_OUT - 1
    for kh in range(KH):
        for kw in range(KW):
            for block in range(ct.cdiv(C_IN, BLOCK_K)):
                cin = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
                cin_valid = cin < C_IN
                x_indices = (((batch[:, None] * H + oh[:, None] + kh) * W + ow[:, None] + kw) * C_IN + cin[None, :])
                safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x)
                x_tile = x_mem.load_offset(safe_x, mask=safe_x >= 0)
                x_tile = x_tile * (row_valid[:, None] & cin_valid[None, :]).astype(torch.float16)
                w_indices = ((kh * KW + kw) * C_IN + cin[:, None]) * C_OUT + cols[None, :]
                safe_w = ct.minimum(ct.maximum(w_indices, 0), max_w)
                w_tile = weight_mem.load_offset(safe_w, mask=cin_valid[:, None] & col_valid[None, :], padding_value=0.0)
                acc = ct.mma(x_tile, w_tile, acc)
    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = rows[:, None] * C_OUT + cols[None, :]
    safe_output = ct.minimum(ct.maximum(output_indices, 0), B * OH * OW * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.conv2d.groups != 1 or self.conv2d.stride != (1, 1) or self.conv2d.padding != (0, 0) or self.conv2d.dilation != (1, 1):
            return torch.nn.functional.conv2d(x.to(torch.float16), self.conv2d.weight.to(torch.float16), None if self.conv2d.bias is None else self.conv2d.bias.to(torch.float16), self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups)
        x = x.to(torch.float16).contiguous(memory_format=torch.channels_last)
        B, C_IN, H, W = x.shape
        KH, KW = self.conv2d.kernel_size
        OH, OW = H - KH + 1, W - KW + 1
        weight = self.conv2d.weight.permute(2, 3, 1, 0).contiguous().to(torch.float16)
        bias = torch.zeros(self.conv2d.out_channels, device=x.device, dtype=torch.float16) if self.conv2d.bias is None else self.conv2d.bias.to(torch.float16).contiguous()
        output = torch.empty((B, self.conv2d.out_channels, OH, OW), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(B * OH * OW, 32), ct.cdiv(self.conv2d.out_channels, 32)), _conv2d_flat_kernel, (x, weight, bias, output, B, C_IN, self.conv2d.out_channels, H, W, OH, OW, KH, KW, 32, 32, 16))
        return output
