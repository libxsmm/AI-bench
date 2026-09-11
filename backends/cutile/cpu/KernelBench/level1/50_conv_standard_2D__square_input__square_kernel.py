# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _conv2d_implicit_gemm(x, weight, bias, output, M: ConstInt, N: ConstInt, K: ConstInt, OH: ConstInt, OW: ConstInt, H: ConstInt, W: ConstInt, BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid = ct.bid(0)
    grid_n = ct.cdiv(N, BLOCK_N)
    group_id = pid // (4 * grid_n)
    group_size = ct.minimum(4, ct.cdiv(M, BLOCK_M) - group_id * 4)
    pid_m = group_id * 4 + (pid % group_size)
    pid_n = (pid % (4 * grid_n)) // group_size
    rows = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    row_valid = rows < M
    col_valid = cols < N
    batch = rows // (OH * OW)
    rem = rows % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    weight_mem = weight.get_raw_memory()
    bias_mem = bias.get_raw_memory()
    max_x = (M // (OH * OW)) * 3 * H * W - 1
    for block in range(ct.cdiv(K, BLOCK_K)):
        offsets_k = block * BLOCK_K + ct.arange(BLOCK_K, dtype=torch.int32)
        kh = offsets_k // (11 * 3)
        kw = (offsets_k // 3) % 11
        cin = offsets_k % 3
        input_h = oh[:, None] * 4 + kh[None, :] - 2
        input_w = ow[:, None] * 4 + kw[None, :] - 2
        valid = row_valid[:, None] & (offsets_k[None, :] < K) & (input_h >= 0) & (input_h < H) & (input_w >= 0) & (input_w < W)
        x_indices = ((batch[:, None] * 3 + cin[None, :]) * H + input_h) * W + input_w
        safe_x = ct.minimum(ct.maximum(x_indices, 0), max_x)
        x_tile = x_mem.load_offset(safe_x, mask=safe_x >= 0)
        x_tile = x_tile * valid.astype(torch.bfloat16)
        weight_indices = offsets_k[:, None] * N + cols[None, :]
        safe_weight = ct.minimum(ct.maximum(weight_indices, 0), K * N - 1)
        weight_tile = weight_mem.load_offset(safe_weight, mask=(offsets_k[:, None] < K) & col_valid[None, :], padding_value=0.0)
        acc = ct.mma(x_tile, weight_tile, acc)
    safe_cols = ct.minimum(ct.maximum(cols, 0), N - 1)
    acc += bias_mem.load_offset(safe_cols, mask=col_valid, padding_value=0.0)[None, :].astype(ct.float32)
    output_indices = ((batch[:, None] * N + cols[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    safe_output = ct.minimum(ct.maximum(output_indices, 0), M * N - 1)
    output.get_raw_memory().store_offset(safe_output, ct.astype(acc, output.dtype), mask=row_valid[:, None] & col_valid[None, :])


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        x = x.to(torch.bfloat16).contiguous()
        B, _, H, W = x.shape
        OH = (H + 4 - 11) // 4 + 1
        OW = (W + 4 - 11) // 4 + 1
        weight = self.conv1.weight.permute(2, 3, 1, 0).contiguous().reshape(-1, 96).to(torch.bfloat16)
        bias = self.conv1.bias.to(torch.bfloat16).contiguous()
        output = torch.empty((B, 96, OH, OW), device=x.device, dtype=torch.bfloat16)
        with cpu.compile_options({"assume_in_bounds": False}):
            ct.launch(None, (ct.cdiv(B * OH * OW, 32) * ct.cdiv(96, 32),), _conv2d_implicit_gemm, (x, weight, bias, output, B * OH * OW, 96, 3 * 11 * 11, OH, OW, H, W, 32, 32, 32))
        return output
