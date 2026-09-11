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
    grid=lambda meta: (meta["B"],),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def huber_row_kernel(pred, target, rows, B: ConstInt, N: ConstInt, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    total = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = row * N + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        diff = (ct.where(valid, ct.gather(pred, offsets), 0.0) - ct.where(valid, ct.gather(target, offsets), 0.0)).astype(ct.float32)
        absolute = ct.abs(diff)
        total += ct.sum(ct.where(absolute < 1.0, 0.5 * diff * diff, absolute - 0.5), axis=0)
    ct.store(rows, index=(row,), tile=total)


@ct.kernel
def mean_kernel(values, output, B: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    total = 0.0
    for block in range(ct.cdiv(B, BLOCK_SIZE)):
        indices = block * BLOCK_SIZE + offsets
        total += ct.sum(ct.where(indices < B, ct.gather(values, indices), 0.0), axis=0)
    ct.store(output, index=(0,), tile=total / B)


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        B, N = predictions.shape
        rows = torch.empty(B, device=predictions.device, dtype=torch.float32)
        output = torch.empty(1, device=predictions.device, dtype=torch.float32)
        huber_row_kernel(None, (predictions.view(-1), targets.view(-1), rows, B, N))
        with cpu.compile_options({"assume_in_bounds": B % 128 == 0}):
            ct.launch(None, (1,), mean_kernel, (rows, output, B, 128))
        return output[0]
