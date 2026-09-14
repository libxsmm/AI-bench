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
def min_reduction_kernel(x, output, B: ConstInt, D1: ConstInt, D2: ConstInt, BLOCK_D1: ConstInt, BLOCK_D2: ConstInt):
    pid_d2 = ct.bid(0)
    batch = ct.bid(1)
    acc = ct.full((BLOCK_D2,), float("inf"), dtype=ct.float32)
    for block in range(ct.cdiv(D1, BLOCK_D1)):
        values = ct.load(
            x,
            (batch, block, pid_d2),
            (1, BLOCK_D1, BLOCK_D2),
            padding_mode=ct.PaddingMode.POS_INF,
        ).reshape((BLOCK_D1, BLOCK_D2)).astype(ct.float32)
        acc = ct.minimum(acc, ct.min(values, axis=0))
    output_tile = ct.astype(acc, output.dtype).reshape((1, BLOCK_D2))
    ct.store(output, (batch, pid_d2), output_tile)


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1, D2 = x.shape
        output = torch.empty((B, D2), device=x.device, dtype=x.dtype)
        BLOCK_D1, BLOCK_D2 = 32, 64
        with cpu.compile_options({"assume_in_bounds": D1 % BLOCK_D1 == 0 and D2 % BLOCK_D2 == 0}):
            ct.launch(None, (ct.cdiv(D2, BLOCK_D2), B), min_reduction_kernel, (x, output, B, D1, D2, BLOCK_D1, BLOCK_D2))
        return output
