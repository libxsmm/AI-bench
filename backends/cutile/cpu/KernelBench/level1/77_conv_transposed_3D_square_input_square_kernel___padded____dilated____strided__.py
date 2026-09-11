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
def _conv_transpose3d_kernel(
    x,
    weight,
    bias,
    output,
    B: ConstInt,
    C_IN: ConstInt,
    C_OUT: ConstInt,
    D: ConstInt,
    H: ConstInt,
    W: ConstInt,
    OD: ConstInt,
    OH: ConstInt,
    OW: ConstInt,
    K: ConstInt,
    D_ACT: ConstInt,
    H_ACT: ConstInt,
    W_ACT: ConstInt,
    BLOCK_W: ConstInt,
    BLOCK_OC: ConstInt,
):
    pid_w = ct.bid(0)
    pid_bdh = ct.bid(1)
    pid_oc = ct.bid(2)
    batch = pid_bdh // (D_ACT * H_ACT)
    rem = pid_bdh % (D_ACT * H_ACT)
    d_idx = rem // H_ACT
    h_idx = rem % H_ACT
    rows = pid_w * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32)
    cols = pid_oc * BLOCK_OC + ct.arange(BLOCK_OC, dtype=torch.int32)
    row_valid = rows < W_ACT
    col_valid = cols < C_OUT
    acc = ct.full((BLOCK_W, BLOCK_OC), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    cin = ct.arange(C_IN, dtype=torch.int32)
    max_x_offset = B * D * H * W * C_IN - 1
    max_weight_offset = K * K * K * C_IN * C_OUT - 1
    for kd in range(K):
        input_d = d_idx + 1 - kd
        valid_d = (input_d >= 0) & (input_d < D)
        for kh in range(K):
            input_h = h_idx + 1 - kh
            valid_h = valid_d & (input_h >= 0) & (input_h < H)
            for kw in range(K):
                input_w = rows + 1 - kw
                valid = row_valid & valid_h & (input_w >= 0) & (input_w < W)
                x_indices = ((((batch * D + input_d) * H + input_h) * W + input_w[:, None]) * C_IN + cin[None, :])
                safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                x_values = x_mem.load_offset(safe_x_indices, mask=safe_x_indices >= 0)
                x_values = x_values * valid[:, None].astype(torch.float16)
                kernel_index = kd * K * K + kh * K + kw
                weight_base = kernel_index * C_IN * C_OUT + cin[:, None] * C_OUT + cols[None, :]
                safe_weight_base = ct.minimum(ct.maximum(weight_base, 0), max_weight_offset)
                weight_values = weight_mem.load_offset(safe_weight_base, mask=col_valid[None, :], padding_value=0.0)
                acc = ct.mma(x_values, weight_values, acc)
    safe_cols = ct.minimum(ct.maximum(cols, 0), C_OUT - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = ((((batch * OD + (2 * d_idx + 1)) * OH + (2 * h_idx + 1)) * OW + (2 * rows[:, None] + 1)) * C_OUT + cols[None, :])
    safe_output_indices = ct.minimum(ct.maximum(output_indices, 0), B * OD * OH * OW * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_output_indices, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False, dilation=(1,1,1)):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous()
        B, C_IN, D, H, W = x.shape
        K = self.conv.kernel_size[0]
        STRIDE = self.conv.stride[0]
        PADDING = self.conv.padding[0]
        DILATION = self.conv.dilation[0]
        OD = (D - 1) * STRIDE - 2 * PADDING + DILATION * (K - 1) + 1
        OH = (H - 1) * STRIDE - 2 * PADDING + DILATION * (K - 1) + 1
        OW = (W - 1) * STRIDE - 2 * PADDING + DILATION * (K - 1) + 1
        C_OUT = self.conv.out_channels
        x_channels_last = x.contiguous(memory_format=torch.channels_last_3d)
        weight = self.conv.weight.permute(2, 3, 4, 0, 1).contiguous().to(dtype=torch.float16)
        bias = torch.zeros(C_OUT, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.zeros((B, C_OUT, OD, OH, OW), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)
        D_ACT = OD // 2
        H_ACT = OH // 2
        W_ACT = OW // 2
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(W_ACT, 16), B * D_ACT * H_ACT, ct.cdiv(C_OUT, 32)), _conv_transpose3d_kernel, (x_channels_last, weight, bias, output, B, C_IN, C_OUT, D, H, W, OD, OH, OW, K, D_ACT, H_ACT, W_ACT, 16, 32))
        return output
