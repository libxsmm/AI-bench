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
    configs=[ct.tune.Config({"BLOCK_N": 32, "BLOCK_K": 64})],
    key=["D1", "D2"],
    grid=lambda meta: (meta["B"], ct.cdiv(meta["D2"], meta["BLOCK_N"])),
    options=lambda meta: {"assume_in_bounds": meta["D1"] % meta["BLOCK_K"] == 0 and meta["D2"] % meta["BLOCK_N"] == 0},
)
@ct.kernel
def max_reduce_dim1_kernel(x, output, B: ConstInt, D1: ConstInt, D2: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    batch = ct.bid(0)
    pid_n = ct.bid(1)
    cols = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=torch.int32)
    acc = ct.full((BLOCK_N,), float("-inf"), dtype=ct.float32)
    x_mem = x.get_raw_memory()
    rows = ct.arange(BLOCK_K, dtype=torch.int32)
    for block in range(ct.cdiv(D1, BLOCK_K)):
        row_ids = block * BLOCK_K + rows
        valid = (row_ids[:, None] < D1) & (cols[None, :] < D2)
        offsets = batch * D1 * D2 + row_ids[:, None] * D2 + cols[None, :]
        safe_offsets = ct.minimum(ct.maximum(offsets, 0), B * D1 * D2 - 1)
        values = x_mem.load_offset(safe_offsets, mask=valid, padding_value=float("-inf")).astype(ct.float32)
        acc = ct.maximum(acc, ct.max(values, axis=0))
    ct.scatter(output, batch * D2 + cols, acc)


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, D1, D2 = x.shape
        output = torch.empty((B, D2), device=x.device, dtype=torch.float32)
        max_reduce_dim1_kernel(None, (x.view(-1), output.view(-1), B, D1, D2))
        return output.to(x.dtype)
