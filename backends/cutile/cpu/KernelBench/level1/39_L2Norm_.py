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
    configs=[ct.tune.Config({"BLOCK_SIZE": 32})],
    key=["N"],
    grid=lambda meta: (meta["M"],),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def l2_norm_kernel(x, output, M: ConstInt, N: ConstInt, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    base = row * N
    total = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = base + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        total += ct.sum(values * values, axis=0)
    inv_norm = 1.0 / ct.sqrt(total + 1e-12)
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = base + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        ct.scatter(output, offsets, ct.astype(values * inv_norm, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, N = x.shape
        output = torch.empty_like(x)
        l2_norm_kernel(None, (x.view(-1), output.view(-1), M, N))
        return output
