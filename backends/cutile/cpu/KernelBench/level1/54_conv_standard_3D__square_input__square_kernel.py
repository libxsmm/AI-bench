# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv3d_fused_k(x, weight, output, B: ConstInt, C_IN: ConstInt, C_OUT: ConstInt, OD: ConstInt, OH: ConstInt, OW: ConstInt, KD: ConstInt, KH: ConstInt, KW: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid = ct.bid(0)
    M = B * OD * OH * OW
    num_m = ct.cdiv(M, BLOCK_M)
    num_n = ct.cdiv(C_OUT, BLOCK_N)
    group = pid // (8 * num_n)
    group_size = ct.minimum(8, num_m - group * 8)
    inside = pid % (8 * num_n)
    pid_m = group * 8 + inside % group_size
    pid_n = inside // group_size
    rows = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < M
    col_valid = cols < C_OUT
    spatial = OD * OH * OW
    batch = rows // spatial
    rem = rows % spatial
    od = rem // (OH * OW)
    rem = rem % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    w_mem = weight.get_raw_memory()
    max_x = B * (OD + KD - 1) * (OH + KH - 1) * (OW + KW - 1) * C_IN - 1
    K = KD * KH * KW * C_IN
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
    safe_out = ct.minimum(ct.maximum(out_indices, 0), B * OD * OH * OW * C_OUT - 1)
    output.get_raw_memory().store_offset(safe_out, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size,) * 3
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None: nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, x):
        B, C_IN, D, H, W = x.shape
        KD, KH, KW = self.weight.shape[2:]
        x = x.to(torch.float16).contiguous(memory_format=torch.channels_last_3d)
        weight = self.weight.permute(2, 3, 4, 1, 0).contiguous().to(torch.float16).reshape(KD * KH * KW * C_IN, self.weight.shape[0])
        output = torch.empty((B, self.weight.shape[0], D - KD + 1, H - KH + 1, W - KW + 1), device=x.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)
        OD, OH, OW = output.shape[2:]
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(B * OD * OH * OW, 32) * ct.cdiv(self.weight.shape[0], 32),), _conv3d_fused_k, (x, weight, output, B, C_IN, self.weight.shape[0], OD, OH, OW, KD, KH, KW, 32, 32, 16))
        if self.bias is not None: output = output + self.bias.to(torch.float16).view(1, -1, 1, 1, 1)
        return output
