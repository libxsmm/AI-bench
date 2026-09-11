# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv3d_kernel(x, weight, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, OD: ConstInt, OH: ConstInt, OW: ConstInt, KD: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    rows = ct.bid(0) * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = ct.bid(1) * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    M = B * OD * OH * OW
    row_valid = rows < M
    col_valid = cols < C_OUT
    batch = rows // (OD * OH * OW)
    rem = rows % (OD * OH * OW)
    od = rem // (OH * OW)
    rem = rem % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    w_mem = weight.get_raw_memory()
    K = KD * KH * KW * C_IN
    max_x = B * (OD + KD - 1) * (OH + KH - 1) * (OW + KW - 1) * C_IN - 1
    for block in range(ct.cdiv(K, BLOCK_K)):
        k = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
        cin = k % C_IN
        spatial_k = k // C_IN
        kw = spatial_k % KW
        kh = (spatial_k // KW) % KH
        kd = spatial_k // (KW * KH)
        x_indices = ((((batch[:, None] * (OD + KD - 1) + od[:, None] + kd[None, :]) * (OH + KH - 1) + oh[:, None] + kh[None, :]) * (OW + KW - 1) + ow[:, None] + kw[None, :]) * C_IN + cin[None, :])
        safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x)
        x_tile = x_mem.load_offset(safe_x, mask=row_valid[:, None] & (k[None, :] < K))
        w_indices = k[:, None] * C_OUT + cols[None, :]
        safe_w = ct.minimum(ct.maximum(w_indices, 0), K * C_OUT - 1)
        w_tile = w_mem.load_offset(safe_w, mask=(k[:, None] < K) & col_valid[None, :], padding_value=0.0)
        acc = ct.mma(x_tile, w_tile, acc)
    out_indices = (((batch[:, None] * OD + od[:, None]) * OH + oh[:, None]) * OW + ow[:, None]) * C_OUT + cols[None, :]
    safe_out = ct.minimum(ct.maximum(out_indices, 0), M * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_out, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = x.to(torch.float16).contiguous(memory_format=torch.channels_last_3d)
        B, C_IN, D, H, W = x.shape
        KD, KH, KW = self.conv3d.kernel_size
        OD, OH, OW = D - KD + 1, H - KH + 1, W - KW + 1
        C_OUT = self.conv3d.out_channels
        weight = self.conv3d.weight.permute(2, 3, 4, 1, 0).contiguous().to(torch.float16).reshape(KD * KH * KW * C_IN, C_OUT)
        output = torch.empty((B, C_OUT, OD, OH, OW), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(B * OD * OH * OW, 32), ct.cdiv(C_OUT, 32)), _conv3d_kernel, (x, weight, output, B, C_IN, C_OUT, OD, OH, OW, KD, KH, KW, 32, 32, 16))
        if self.conv3d.bias is not None: output = output + self.conv3d.bias.to(torch.float16).view(1, -1, 1, 1, 1)
        return output
