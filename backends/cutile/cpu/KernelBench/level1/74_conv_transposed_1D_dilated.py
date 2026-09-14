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
def _conv_transpose1d_dilated_kernel(x, weight, bias, output, B: ConstInt, L_IN: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, L_OUT: ConstInt, K_SIZE: ConstInt, DILATION: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid = ct.bid(0)
    num_n = ct.cdiv(C_OUT, BLOCK_N)
    num_m = ct.cdiv(L_OUT, BLOCK_M)
    pid_m = pid // num_n
    pid_n = pid % num_n
    batch = pid_m // num_m
    block_m = pid_m % num_m
    row_start = block_m * BLOCK_M
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_view = x.tiled_view(
        (1, BLOCK_M, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=(1, 1, BLOCK_K),
    )
    weight_view = weight.tiled_view(
        (1, BLOCK_K, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=(1, BLOCK_K, BLOCK_N),
    )
    x_mem = x.get_raw_memory()
    max_x_offset = B * L_IN * C_IN - 1
    cin = ct.arange(BLOCK_K, dtype=torch.int32)
    valid_c = cin < C_IN
    for k in range(K_SIZE):
        input_row_start = row_start - k * DILATION
        input_rows = row_start + ct.arange(BLOCK_M, dtype=torch.int32) - k * DILATION
        valid_rows = (input_rows >= 0) & (input_rows < L_IN)
        if input_row_start >= 0:
            safe_input_row_start = ct.minimum(input_row_start, L_IN - 1)
            x_values = x_view.load(
                (batch, safe_input_row_start, 0),
            ).reshape((BLOCK_M, BLOCK_K)).astype(ct.float32)
            x_values = x_values * valid_rows[:, None].astype(torch.float32)
        else:
            x_indices = (
                ((batch * L_IN + input_rows[:, None]) * C_IN)
                + cin[None, :]
            )
            safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
            x_values = x_mem.load_offset(
                safe_x_indices,
                mask=valid_rows[:, None] & valid_c[None, :],
                padding_value=0.0,
            ).astype(ct.float32)
        weight_values = weight_view.load(
            (k, 0, pid_n),
        ).reshape((BLOCK_K, BLOCK_N)).astype(ct.float32)
        acc = ct.mma(x_values, weight_values, acc)
    bias_values = ct.load(
        bias,
        (pid_n,),
        (BLOCK_N,),
        padding_mode=ct.PaddingMode.ZERO,
    ).astype(ct.float32)
    acc += bias_values[None, :]
    output_view = output.tiled_view((1, BLOCK_M, BLOCK_N))
    output_tile = ct.astype(acc, output.dtype).reshape((1, BLOCK_M, BLOCK_N))
    output_view.store((batch, block_m, pid_n), output_tile)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.contiguous()
        B, C_IN, L_IN = x.shape
        K_SIZE = self.conv.kernel_size[0]
        DILATION = self.conv.dilation[0]
        L_OUT = L_IN + DILATION * (K_SIZE - 1)
        weight = self.conv.weight.permute(2, 0, 1).contiguous().to(dtype=x.dtype)
        bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=x.dtype) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=x.dtype)
        output = torch.empty((B, L_OUT, self.conv.out_channels), device=x.device, dtype=x.dtype)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(L_OUT, 32) * ct.cdiv(self.conv.out_channels, 32) * B,), _conv_transpose1d_dilated_kernel, (x.permute(0, 2, 1).contiguous(), weight, bias, output, B, L_IN, C_IN, self.conv.out_channels, L_OUT, K_SIZE, DILATION, 32, 32, 16))
        return output.permute(0, 2, 1).contiguous()
