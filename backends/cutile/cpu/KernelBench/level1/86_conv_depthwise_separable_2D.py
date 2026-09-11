# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn
import torch.nn.functional as F

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv2d_fused(x, weight, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, H: ConstInt, W: ConstInt, OH: ConstInt, OW: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    rows = ct.bid(0) * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = ct.bid(1) * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    M = B * OH * OW
    row_valid = rows < M
    col_valid = cols < C_OUT
    batch = rows // (OH * OW)
    rem = rows % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory(); w_mem = weight.get_raw_memory()
    max_x = B * H * W * C_IN - 1
    for kh in range(KH):
        for kw in range(KW):
            for block in range(ct.cdiv(C_IN, BLOCK_K)):
                cin = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
                cin_valid = cin < C_IN
                x_indices = (((batch[:, None] * H + oh[:, None] + kh) * W + ow[:, None] + kw) * C_IN + cin[None, :])
                safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x)
                x_tile = x_mem.load_offset(safe_x, mask=safe_x >= 0)
                x_tile = x_tile * (row_valid[:, None] & cin_valid[None, :]).astype(torch.float32)
                w_indices = ((kh * KW + kw) * C_IN + cin[:, None]) * C_OUT + cols[None, :]
                safe_w = ct.minimum(ct.maximum(w_indices, 0), KH * KW * C_IN * C_OUT - 1)
                w_tile = w_mem.load_offset(safe_w, mask=cin_valid[:, None] & col_valid[None, :], padding_value=0.0)
                acc = ct.mma(x_tile, w_tile, acc)
    output_indices = ((batch[:, None] * OH + oh[:, None]) * OW + ow[:, None]) * C_OUT + cols[None, :]
    safe_out = ct.minimum(ct.maximum(output_indices, 0), M * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_out, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self._padding = int(padding)

    def forward(self, x):
        x = x.to(torch.float32)
        B, C_IN, H, W = x.shape
        KH, KW = self.depthwise.kernel_size
        if self._padding:
            x = F.pad(x, (self._padding,) * 4)
        H_PAD, W_PAD = x.shape[2:]
        OH, OW = H_PAD - KH + 1, W_PAD - KW + 1
        C_OUT = self.pointwise.out_channels
        combined = (self.pointwise.weight * self.depthwise.weight.transpose(0, 1)).to(torch.float32).permute(2, 3, 1, 0).contiguous()
        x_nhwc = x.contiguous(memory_format=torch.channels_last).permute(0, 2, 3, 1).contiguous()
        output = torch.empty((B, C_OUT, OH, OW), device=x.device, dtype=torch.float32).contiguous(memory_format=torch.channels_last)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(B * OH * OW, 32), ct.cdiv(C_OUT, 32)), _conv2d_fused, (x_nhwc, combined, output, B, C_IN, C_OUT, H_PAD, W_PAD, OH, OW, KH, KW, 32, 32, 16))
        return output
