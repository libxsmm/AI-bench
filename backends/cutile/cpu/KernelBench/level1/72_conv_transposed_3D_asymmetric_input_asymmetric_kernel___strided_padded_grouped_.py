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
def _conv_transpose3d_grouped_kernel(x, weight, bias, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, D_IN: ConstInt, H_IN: ConstInt, W_IN: ConstInt, D_OUT: ConstInt, H_OUT: ConstInt, W_OUT: ConstInt, C_IN_PG: ConstInt, C_OUT_PG: ConstInt, GROUPS: ConstInt, KD: ConstInt, KH: ConstInt, KW: ConstInt, STRIDE_D: ConstInt, STRIDE_H: ConstInt, STRIDE_W: ConstInt, PAD_D: ConstInt, PAD_H: ConstInt, PAD_W: ConstInt, BLOCK_W: ConstInt, BLOCK_OH: ConstInt):
    pid_bg = ct.bid(0); pid_d = ct.bid(1); pid_hw = ct.bid(2)
    batch = pid_bg // GROUPS; group = pid_bg % GROUPS; d_out = pid_d
    num_w = ct.cdiv(W_OUT, BLOCK_W); oh0 = (pid_hw // num_w) * BLOCK_OH; ow0 = (pid_hw % num_w) * BLOCK_W
    rows = ow0 + ct.arange(BLOCK_W, dtype=torch.int32); cols = ct.arange(C_OUT_PG, dtype=torch.int32)
    row_valid = rows < W_OUT
    x_mem = x.get_raw_memory(); weight_mem = weight.get_raw_memory(); output_mem = output.get_raw_memory(); bias_mem = bias.get_raw_memory()
    max_x_offset = B * C_IN * D_IN * H_IN * W_IN - 1; max_weight_offset = C_IN * KD * KH * KW * C_OUT_PG - 1
    for oh_local in range(BLOCK_OH):
        oh = oh0 + oh_local
        valid_oh = oh < H_OUT
        acc = ct.full((BLOCK_W, C_OUT_PG), 0.0, dtype=ct.float32)
        for kd in range(KD):
            input_d_num = d_out + PAD_D - kd
            input_d = input_d_num // STRIDE_D
            valid_d = (input_d_num % STRIDE_D == 0) & (input_d_num >= 0) & (input_d < D_IN)
            for kh in range(KH):
                input_h_num = oh + PAD_H - kh
                input_h = input_h_num // STRIDE_H
                valid_h = valid_oh & (input_h_num % STRIDE_H == 0) & (input_h_num >= 0) & (input_h < H_IN)
                for kw in range(KW):
                    input_w_num = rows + PAD_W - kw
                    input_w = input_w_num // STRIDE_W
                    valid = row_valid & valid_d & valid_h & (input_w_num % STRIDE_W == 0) & (input_w_num >= 0) & (input_w < W_IN)
                    cin = ct.arange(C_IN_PG, dtype=torch.int32)
                    x_indices = (((batch * C_IN + group * C_IN_PG + cin[None, :]) * D_IN + input_d) * H_IN + input_h) * W_IN + input_w[:, None]
                    safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                    x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0) * valid[:, None].astype(torch.float16)
                    kernel_index = ((kd * KH + kh) * KW + kw) * C_OUT_PG
                    weight_base = group * max_weight_offset + kernel_index + cin[:, None] * KD * KH * KW * C_OUT_PG + cols[None, :]
                    safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), C_IN * KD * KH * KW * C_OUT_PG * GROUPS - 1)
                    weight_values = weight_mem.load_offset(safe_weight_base, mask=cols[None, :] < C_OUT_PG, padding_value=0.0)
                    acc = ct.mma(x_values, weight_values, acc)
        output_cols = group * C_OUT_PG + cols
        output_indices = (((batch * C_OUT + output_cols[None, :]) * D_OUT + d_out) * H_OUT + oh) * W_OUT + rows[:, None]
        safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), B * C_OUT * D_OUT * H_OUT * W_OUT - 1)
        output_mem.store_offset(safe_output_indices, ct.astype(acc, output.dtype), mask=row_valid[:, None] & valid_oh)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False, dilation=(1,1,1)):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous(); B, C_IN, D_IN, H_IN, W_IN = x.shape
        KD, KH, KW = self.conv.kernel_size; SD, SH, SW = self.conv.stride; PD, PH, PW = self.conv.padding; OPD, OPH, OPW = self.conv.output_padding
        D_OUT = (D_IN - 1) * SD - 2 * PD + KD + OPD; H_OUT = (H_IN - 1) * SH - 2 * PH + KH + OPH; W_OUT = (W_IN - 1) * SW - 2 * PW + KW + OPW
        C_IN_PG = C_IN // self.conv.groups; C_OUT_PG = self.conv.out_channels // self.conv.groups
        weight = self.conv.weight.permute(0, 2, 3, 4, 1).contiguous().to(dtype=torch.float16)
        bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.empty((B, self.conv.out_channels, D_OUT, H_OUT, W_OUT), device=x.device, dtype=torch.float16)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (B * self.conv.groups, D_OUT, ct.cdiv(H_OUT, 2) * ct.cdiv(W_OUT, 64)), _conv_transpose3d_grouped_kernel, (x, weight, bias, output, B, C_IN, self.conv.out_channels, D_IN, H_IN, W_IN, D_OUT, H_OUT, W_OUT, C_IN_PG, C_OUT_PG, self.conv.groups, KD, KH, KW, SD, SH, SW, PD, PH, PW, 64, 2))
        return output
