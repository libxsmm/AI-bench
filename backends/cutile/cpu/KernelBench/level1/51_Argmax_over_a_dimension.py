# ruff: noqa: E731
# Example CUDA Tile CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def argmax_dim1_kernel(x, output, B: ConstInt, D1: ConstInt, D2: ConstInt, BLOCK_N: ConstInt):
    pid_n = ct.bid(0)
    batch = ct.bid(1)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    valid = cols < D2
    max_value = ct.full((BLOCK_N,), float("-inf"), dtype=ct.float32)
    max_index = ct.full((BLOCK_N,), 0, dtype=torch.int32)
    for k in range(D1):
        offsets = batch * D1 * D2 + k * D2 + cols
        values = ct.where(valid, ct.gather(x, offsets), float("-inf")).astype(ct.float32)
        update = values > max_value
        max_value = ct.where(update, values, max_value)
        max_index = ct.where(update, k, max_index)
    ct.scatter(output, batch * D2 + cols, ct.astype(max_index, ct.int64))


class Model(nn.Module):
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1, D2 = x.shape
        output = torch.empty((B, D2), device=x.device, dtype=torch.int64)
        BLOCK_N = 32
        with cpu.compile_options({"assume_in_bounds": D2 % BLOCK_N == 0}):
            ct.launch(None, (ct.cdiv(D2, BLOCK_N), B), argmax_dim1_kernel, (x.view(-1), output.view(-1), B, D1, D2, BLOCK_N))
        return output
