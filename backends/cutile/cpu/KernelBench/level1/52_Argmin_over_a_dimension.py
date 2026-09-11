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
def argmin_kernel(x, output, B: ConstInt, D1: ConstInt, D2: ConstInt, BLOCK_D2: ConstInt):
    pid = ct.bid(0)
    blocks = ct.cdiv(D2, BLOCK_D2)
    batch = pid // blocks
    block = pid % blocks
    cols = block * BLOCK_D2 + ct.arange(BLOCK_D2, dtype=torch.int32)
    valid = cols < D2
    min_value = ct.full((BLOCK_D2,), float("inf"), dtype=ct.float32)
    min_index = ct.full((BLOCK_D2,), 0, dtype=torch.int32)
    for k in range(D1):
        offsets = batch * D1 * D2 + k * D2 + cols
        values = ct.where(valid, ct.gather(x, offsets), float("inf")).astype(ct.float32)
        update = values < min_value
        min_value = ct.where(update, values, min_value)
        min_index = ct.where(update, k, min_index)
    ct.scatter(output, batch * D2 + cols, ct.astype(min_index, ct.int64))


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1, D2 = x.shape
        output = torch.empty((B, D2), device=x.device, dtype=torch.int64)
        BLOCK_D2 = 128
        with cpu.compile_options({"assume_in_bounds": D2 % BLOCK_D2 == 0}):
            ct.launch(None, (B * ct.cdiv(D2, BLOCK_D2),), argmin_kernel, (x.view(-1), output.view(-1), B, D1, D2, BLOCK_D2))
        return output
