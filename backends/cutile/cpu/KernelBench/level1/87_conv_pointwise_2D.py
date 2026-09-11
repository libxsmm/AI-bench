# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _pointwise_gemm(x, weight, bias, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, HW: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid = ct.bid(0)
    batch = ct.bid(1)
    num_m = ct.cdiv(C_OUT, BLOCK_M)
    group = pid // (8 * ct.cdiv(HW, BLOCK_N))
    group_size = ct.minimum(8, num_m - group * 8)
    inside = pid % (8 * ct.cdiv(HW, BLOCK_N))
    pid_m = group * 8 + inside % group_size
    pid_n = inside // group_size
    cols = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    spatial = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    col_valid = cols < C_OUT
    spatial_valid = spatial < HW
    acc = ct.full((BLOCK_N, BLOCK_M), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory(); w_mem = weight.get_raw_memory()
    max_x = B * C_IN * HW - 1
    for block in range(ct.cdiv(C_IN, BLOCK_K)):
        cin = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
        cin_valid = cin < C_IN
        x_indices = ((batch * C_IN + cin[None, :]) * HW + spatial[:, None])
        safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x)
        x_tile = x_mem.load_offset(safe_x, mask=spatial_valid[:, None] & cin_valid[None, :])
        w_indices = cols[None, :] * C_IN + cin[:, None]
        safe_w = ct.minimum(ct.maximum(w_indices, 0), C_IN * C_OUT - 1)
        w_tile = w_mem.load_offset(safe_w, mask=cin_valid[:, None] & col_valid[None, :], padding_value=0.0)
        acc = ct.mma(x_tile, w_tile, acc)
    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias.get_raw_memory().load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = batch * C_OUT * HW + cols[None, :] * HW + spatial[:, None]
    safe_output = ct.minimum(ct.maximum(output_indices, 0), B * C_OUT * HW - 1)
    output.get_raw_memory().store_offset(safe_output, ct.astype(acc, output.dtype), mask=spatial_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C_IN, H, W = x.shape
        C_OUT = self.conv1d.out_channels
        HW = H * W
        weight = self.conv1d.weight.squeeze(-1).squeeze(-1).contiguous().to(x.dtype)
        bias = torch.zeros(C_OUT, device=x.device, dtype=x.dtype) if self.conv1d.bias is None else self.conv1d.bias.contiguous().to(x.dtype)
        output = torch.empty((B, C_OUT, H, W), device=x.device, dtype=x.dtype)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(C_OUT, 32) * ct.cdiv(HW, 32), B), _pointwise_gemm, (x.contiguous(), weight, bias, output, B, C_IN, C_OUT, HW, 32, 32, 16))
        return output
