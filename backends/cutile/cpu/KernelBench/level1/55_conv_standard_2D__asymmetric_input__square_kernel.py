# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv2d_kernel(
    x,
    weight,
    bias,
    output,
    B: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    C_IN_PG: ConstInt,
    C_OUT_PG: ConstInt,
    GROUPS: ConstInt,
    H: ConstInt,
    W: ConstInt,
    OH: ConstInt,
    OW: ConstInt,
    KH: ConstInt,
    KW: ConstInt,
    STRIDE_H: ConstInt,
    STRIDE_W: ConstInt,
    PAD_H: ConstInt,
    PAD_W: ConstInt,
    DILATION_H: ConstInt,
    DILATION_W: ConstInt,
    BLOCK_OW: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    batch = ct.bid(0)
    oh = ct.bid(1)
    packed_pid = ct.bid(2)
    num_ow = ct.cdiv(OW, BLOCK_OW)
    num_n = ct.cdiv(C_OUT_PG, BLOCK_N)
    group = packed_pid // (num_ow * num_n)
    local_pid = packed_pid % (num_ow * num_n)
    pid_ow = local_pid % num_ow
    pid_n = local_pid // num_ow
    ow = pid_ow * BLOCK_OW + ct.arange(BLOCK_OW, dtype=torch.int32)
    output_local = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    cols = group * C_OUT_PG + output_local
    row_valid = ow < OW
    col_valid = output_local < C_OUT_PG
    acc = ct.full((BLOCK_OW, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    max_x_offset = B * H * W * C_IN - 1
    max_weight_offset = GROUPS * KH * KW * C_IN_PG * C_OUT_PG - 1

    for kh in range(KH):
        input_h = oh * STRIDE_H + kh * DILATION_H - PAD_H
        valid_h = (input_h >= 0) & (input_h < H)
        for kw in range(KW):
            input_w = ow * STRIDE_W + kw * DILATION_W - PAD_W
            valid = row_valid & valid_h & (input_w >= 0) & (input_w < W)
            for block in range(ct.cdiv(C_IN_PG, BLOCK_K)):
                cin = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
                valid_c = cin < C_IN_PG
                x_indices = (((batch * H + input_h) * W + input_w[:, None]) * C_IN + group * C_IN_PG + cin[None, :])
                safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0)
                x_values = x_values * (valid[:, None] & valid_c[None, :]).astype(x.dtype)
                weight_indices = (group * KH * KW * C_IN_PG * C_OUT_PG + (kh * KW + kw) * C_IN_PG * C_OUT_PG + cin[:, None] * C_OUT_PG + output_local[None, :])
                safe_weight_indices = ct.minimum(ct.maximum(weight_indices, 0), max_weight_offset)
                weight_values = weight_mem.load_offset(safe_weight_indices, mask=valid_c[:, None] & col_valid[None, :], padding_value=0.0)
                acc = ct.mma(x_values, weight_values, acc)

    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = ((batch * OH + oh) * OW + ow[:, None]) * C_OUT + cols[None, :]
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), B * OH * OW * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output_indices, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = x.to(dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        B, C_IN, H, W = x.shape
        KH, KW = self.conv2d.kernel_size
        STRIDE_H, STRIDE_W = self.conv2d.stride
        PAD_H, PAD_W = self.conv2d.padding
        DILATION_H, DILATION_W = self.conv2d.dilation
        OH = (H + 2 * PAD_H - DILATION_H * (KH - 1) - 1) // STRIDE_H + 1
        OW = (W + 2 * PAD_W - DILATION_W * (KW - 1) - 1) // STRIDE_W + 1
        C_OUT = self.conv2d.out_channels
        C_IN_PG = C_IN // self.conv2d.groups
        C_OUT_PG = C_OUT // self.conv2d.groups
        weight = self.conv2d.weight.reshape(self.conv2d.groups, C_OUT_PG, C_IN_PG, KH, KW).permute(0, 3, 4, 2, 1).contiguous().to(dtype=torch.bfloat16)
        bias = torch.zeros(C_OUT, device=x.device, dtype=torch.bfloat16) if self.conv2d.bias is None else self.conv2d.bias.contiguous().to(dtype=torch.bfloat16)
        output = torch.empty((B, C_OUT, OH, OW), device=x.device, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (B, OH, self.conv2d.groups * ct.cdiv(OW, 64) * ct.cdiv(C_OUT_PG, 32)), _conv2d_kernel, (x, weight, bias, output, B, C_IN, C_OUT, C_IN_PG, C_OUT_PG, self.conv2d.groups, H, W, OH, OW, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, DILATION_H, DILATION_W, 64, 32, 32))
        return output
