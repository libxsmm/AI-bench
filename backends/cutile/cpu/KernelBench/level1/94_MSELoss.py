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
    grid=lambda meta: (meta["B"],),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def mse_row_kernel(pred, target, rows, B: ConstInt, N: ConstInt, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    total = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = row * N + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        diff = (ct.where(valid, ct.gather(pred, offsets), 0.0) - ct.where(valid, ct.gather(target, offsets), 0.0)).astype(ct.float32)
        total += ct.sum(diff * diff, axis=0)
    ct.store(rows, index=(row,), tile=total)


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        B, N = predictions.shape
        rows = torch.empty(B, device=predictions.device, dtype=torch.float32)
        mse_row_kernel(None, (predictions.view(-1), targets.view(-1), rows, B, N))
        return torch.sum(rows) / (B * N)
