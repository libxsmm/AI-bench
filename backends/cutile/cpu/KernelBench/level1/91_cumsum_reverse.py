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
    configs=[ct.tune.Config({"BLOCK_SIZE": 32}, num_worker_warps=4)],
    key=["N"],
    grid=lambda meta: (meta["M"],),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def reverse_cumsum_kernel(x, output, M: ConstInt, N: ConstInt, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    total = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = row * N + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        total += ct.sum(ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32), axis=0)
    prefix = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = row * N + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        scanned = ct.cumsum(values, axis=0)
        ct.scatter(output, offsets, total - prefix - scanned + values)
        prefix += ct.sum(values, axis=0)


class Model(nn.Module):
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.contiguous()
        M, N = x.shape
        output = torch.empty_like(x)
        reverse_cumsum_kernel(None, (x.view(-1), output.view(-1), M, N))
        return output
