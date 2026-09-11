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
    N: ConstInt,
    H_PAD: ConstInt,
    W_PAD: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    OH: ConstInt,
    OW: ConstInt,
    KH: ConstInt,
    KW: ConstInt,
    BLOCK_OW: ConstInt,
    BLOCK_N: ConstInt,
):
    pid = ct.bid(0)
    batch = ct.bid(1)
    num_ow = ct.cdiv(OW, BLOCK_OW)
    num_n = ct.cdiv(C_OUT, BLOCK_N)
    m = pid // num_n
    pid_n = pid % num_n
    oh = m // num_ow
    pid_ow = m % num_ow
    ow = pid_ow * BLOCK_OW + ct.arange(BLOCK_OW, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = ow < OW
    col_valid = cols < C_OUT
    acc = ct.full((BLOCK_OW, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    cin = ct.arange(C_IN, dtype=torch.int32)
    max_x_offset = N * H_PAD * W_PAD * C_IN - 1
    max_weight_offset = KH * KW * C_IN * C_OUT - 1

    for kh in range(KH):
        for kw in range(KW):
            input_w = ow + kw
            valid = row_valid
            x_indices = ((((batch * H_PAD + oh + kh) * W_PAD + input_w[:, None]) * C_IN) + cin[None, :])
            safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
            x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0)
            x_values = x_values * valid[:, None].astype(torch.float16)
            weight_base = (kh * KW + kw) * C_IN * C_OUT + cin[:, None] * C_OUT + cols[None, :]
            safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), max_weight_offset)
            weight_values = weight_mem.load_offset(safe_weight_base, mask=col_valid[None, :], padding_value=0.0)
            acc = ct.mma(x_values, weight_values, acc)

    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = ((batch * OH + oh) * OW + ow[:, None]) * C_OUT + cols[None, :]
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), N * OH * OW * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output_indices, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16)
        N, C_IN, H, W = x.shape
        KH, KW = self.conv.kernel_size
        OH = H + KH - 1
        OW = W + KW - 1
        x_padded = torch.nn.functional.pad(x, (KW - 1, KW - 1, KH - 1, KH - 1)).contiguous(memory_format=torch.channels_last)
        weight = self.conv.weight.flip(2, 3).permute(2, 3, 0, 1).contiguous().to(dtype=torch.float16)
        bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.empty((N, self.conv.out_channels, OH, OW), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (OH * ct.cdiv(OW, 64) * ct.cdiv(self.conv.out_channels, 32), N), _conv_transpose2d_kernel, (x_padded, weight, bias, output, N, H + 2 * (KH - 1), W + 2 * (KW - 1), C_IN, self.conv.out_channels, OH, OW, KH, KW, 64, 32))
        return output
