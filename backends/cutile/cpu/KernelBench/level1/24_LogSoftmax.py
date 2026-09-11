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


@ct.autotune(
    configs=[ct.tune.Config({"BLOCK_N": size}) for size in [32, 64, 128, 256, 512, 1024, 2048, 4096]],
    key=["N"],
    grid=lambda meta: (meta["M"],),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_N"] == 0},
)
@ct.kernel
def _logsoftmax_kernel(x, output, M: ConstInt, N: ConstInt, BLOCK_N: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_N, dtype=torch.int32)
    base = row * N
    max_value = float("-inf")
    total = 0.0
    for block in range(ct.cdiv(N, BLOCK_N)):
        offsets = base + block * BLOCK_N + cols
        valid = block * BLOCK_N + cols < N
        values = ct.where(valid, ct.gather(x, offsets), float("-inf"))
        block_max = ct.max(values, axis=0)
        new_max = ct.maximum(max_value, block_max)
        total = total * ct.exp2((max_value - new_max) * 1.4426950408889634) + ct.sum(ct.exp2((values - new_max) * 1.4426950408889634), axis=0)
        max_value = new_max
    log_sum = ct.log(total)
    for block in range(ct.cdiv(N, BLOCK_N)):
        offsets = base + block * BLOCK_N + cols
        valid = block * BLOCK_N + cols < N
        values = ct.where(valid, ct.gather(x, offsets), float("-inf"))
        ct.scatter(output, offsets, values - max_value - log_sum)


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, N = x.shape
        output = torch.empty_like(x)
        _logsoftmax_kernel(None, (x.view(-1), output.view(-1), M, N))
        return output
