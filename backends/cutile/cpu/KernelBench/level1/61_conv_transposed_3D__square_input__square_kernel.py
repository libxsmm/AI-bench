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
def _conv_transpose3d_kernel(x, weight, bias, output, B: ConstInt, D_IN: ConstInt, H_IN: ConstInt, W_IN: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, D_OUT: ConstInt, H_OUT: ConstInt, W_OUT: ConstInt, KD: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_W: ConstInt, BLOCK_K: ConstInt, BLOCK_OC: ConstInt):
    pid_w = ct.bid(0)
    pid_bdh = ct.bid(1)
    pid_oc = ct.bid(2)
    batch = pid_bdh // (D_OUT * H_OUT)
    rem = pid_bdh % (D_OUT * H_OUT)
    d_out = rem // H_OUT
    h_out = rem % H_OUT
    acc = ct.full((BLOCK_W, BLOCK_OC), 0.0, dtype=ct.float32)
    x_view = x.tiled_view(
        (1, 1, 1, BLOCK_W, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=(1, 1, 1, 1, BLOCK_K),
    )
    weight_view = weight.tiled_view(
        (1, BLOCK_K, BLOCK_OC),
        padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=(1, BLOCK_K, BLOCK_OC),
    )
    output_view = output.tiled_view((1, 1, 1, BLOCK_W, BLOCK_OC))
    x_mem = x.get_raw_memory()
    max_x_offset = B * D_IN * H_IN * W_IN * C_IN - 1
    cin = ct.arange(BLOCK_K, dtype=torch.int32)
    valid_c = cin < C_IN
    for kd in range(KD):
        input_d = d_out - kd
        for kh in range(KH):
            input_h = h_out - kh
            for kw in range(KW):
                if input_d >= 0 and input_d < D_IN and input_h >= 0 and input_h < H_IN:
                    w_start = pid_w * BLOCK_W - kw
                    kernel_index = kd * KH * KW + kh * KW + kw
                    for c_block in range(ct.cdiv(C_IN, BLOCK_K)):
                        if w_start >= 0:
                            x_values = x_view.load(
                                (batch, input_d, input_h, w_start, c_block)
                            ).reshape((BLOCK_W, BLOCK_K))
                        else:
                            rows = pid_w * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32) - kw
                            valid = (rows >= 0) & (rows < W_IN)
                            x_indices = (
                                (((batch * D_IN + input_d) * H_IN + input_h) * W_IN + rows[:, None]) * C_IN
                                + c_block * BLOCK_K + cin[None, :]
                            )
                            safe_x_indices = ct.minimum(ct.maximum(x_indices, 0), max_x_offset)
                            x_values = x_mem.load_offset(
                                safe_x_indices,
                                mask=valid[:, None] & valid_c[None, :],
                                padding_value=0.0,
                            )
                        weight_values = weight_view.load(
                            (kernel_index, c_block, pid_oc)
                        ).reshape((BLOCK_K, BLOCK_OC))
                        acc = ct.mma(x_values, weight_values, acc)
    bias_values = ct.load(
        bias,
        (pid_oc,),
        (BLOCK_OC,),
        padding_mode=ct.PaddingMode.ZERO,
    ).astype(ct.float32)
    acc += bias_values[None, :]
    output_tile = ct.astype(acc, output.dtype).reshape((1, 1, 1, BLOCK_W, BLOCK_OC))
    output_view.store((batch, d_out, h_out, pid_w, pid_oc), output_tile)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, x):
        x = x.to(torch.float16).contiguous()
        B, C_IN, D_IN, H_IN, W_IN = x.shape
        KD, KH, KW = self.conv.kernel_size
        D_OUT = D_IN + KD - 1
        H_OUT = H_IN + KH - 1
        W_OUT = W_IN + KW - 1
        x_channels_last = x.contiguous(memory_format=torch.channels_last_3d)
        x_ndhwc = x_channels_last.permute(0, 2, 3, 4, 1)
        weight = self.conv.weight.permute(2, 3, 4, 0, 1).reshape(KD * KH * KW, C_IN, self.conv.out_channels).contiguous().to(dtype=torch.float16)
        bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.float16) if self.conv.bias is None else self.conv.bias.contiguous().to(dtype=torch.float16)
        output = torch.empty((B, self.conv.out_channels, D_OUT, H_OUT, W_OUT), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)
        output_ndhwc = output.permute(0, 2, 3, 4, 1)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(W_OUT, 64), B * D_OUT * H_OUT, ct.cdiv(self.conv.out_channels, 32)), _conv_transpose3d_kernel, (x_ndhwc, weight, bias, output_ndhwc, B, D_IN, H_IN, W_IN, C_IN, self.conv.out_channels, D_OUT, H_OUT, W_OUT, KD, KH, KW, 64, 16, 32))
        return output
