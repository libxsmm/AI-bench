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
    configs=[ct.tune.Config({"BLOCK_N": 32})],
    key=["N"],
    grid=lambda meta: (meta["B"],),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_N"] == 0},
)
@ct.kernel
def cross_entropy_kernel(logits, targets, losses, B: ConstInt, N: ConstInt, BLOCK_N: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_N, dtype=torch.int32)
    maximum = float("-inf")
    total = 0.0
    for block in range(ct.cdiv(N, BLOCK_N)):
        offsets = row * N + block * BLOCK_N + cols
        valid = block * BLOCK_N + cols < N
        values = ct.where(valid, ct.gather(logits, offsets), float("-inf")).astype(ct.float32)
        block_max = ct.max(values, axis=0)
        new_max = ct.maximum(maximum, block_max)
        total = total * ct.exp2((maximum - new_max) * 1.4426950408889634) + ct.sum(ct.exp2((values - new_max) * 1.4426950408889634), axis=0)
        maximum = new_max
    log_sum = ct.log(total)
    target = ct.load(targets, index=(row,), shape=())
    target_logit = ct.load(logits, index=(row * N + target,), shape=()).astype(ct.float32)
    ct.store(losses, index=(row,), tile=-target_logit + maximum + log_sum)


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
        losses = torch.empty(B, device=predictions.device, dtype=torch.float32)
        output = torch.empty(1, device=predictions.device, dtype=torch.float32)
        cross_entropy_kernel(None, (predictions.view(-1), targets.view(-1), losses, B, N))
        with cpu.compile_options({"assume_in_bounds": B % 128 == 0}):
            ct.launch(None, (1,), mean_kernel, (losses, output, B, 128))
        return output[0]
