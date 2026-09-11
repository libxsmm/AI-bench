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
    configs=[ct.tune.Config({"BLOCK_R": 32, "BLOCK_N": 64})],
    key=["R", "C"],
    grid=lambda meta: (ct.cdiv(meta["C"], meta["BLOCK_N"]), meta["B"]),
    options=lambda meta: {"assume_in_bounds": meta["R"] % meta["BLOCK_R"] == 0 and meta["C"] % meta["BLOCK_N"] == 0},
)
@ct.kernel
def mean_reduce_kernel(x, output, B: ConstInt, R: ConstInt, C: ConstInt, BLOCK_R: ConstInt, BLOCK_N: ConstInt):
    pid_n = ct.bid(0)
    pid_b = ct.bid(1)
    rows = ct.arange(BLOCK_R, dtype=torch.int32)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)
    x_mem = x.get_raw_memory()
    for block in range(ct.cdiv(R, BLOCK_R)):
        row_ids = block * BLOCK_R + rows
        valid = (row_ids[:, None] < R) & (cols[None, :] < C)
        offsets = pid_b * R * C + row_ids[:, None] * C + cols[None, :]
        safe_offsets = ct.minimum(ct.maximum(offsets, 0), B * R * C - 1)
        values = x_mem.load_offset(safe_offsets, mask=valid, padding_value=0.0).astype(ct.float32)
        acc += ct.sum(values, axis=0)
    ct.scatter(output, pid_b * C + cols, ct.astype(acc / R, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.dim == 1
        x = x.contiguous()
        B, R, C = x.shape
        output = torch.empty((B, C), device=x.device, dtype=x.dtype)
        mean_reduce_kernel(None, (x.view(-1), output.view(-1), B, R, C))
        return output
