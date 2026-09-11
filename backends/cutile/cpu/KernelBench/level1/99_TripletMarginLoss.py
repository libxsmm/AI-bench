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
    configs=[ct.tune.Config({"BLOCK_K": 32})],
    key=["D"],
    grid=lambda meta: (meta["B"],),
    options=lambda meta: {"assume_in_bounds": meta["D"] % meta["BLOCK_K"] == 0},
)
@ct.kernel
def triplet_kernel(anchor, positive, negative, rows, B: ConstInt, D: ConstInt, margin, eps, BLOCK_K: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_K, dtype=torch.int32)
    positive_total = 0.0
    negative_total = 0.0
    for block in range(ct.cdiv(D, BLOCK_K)):
        offsets = row * D + block * BLOCK_K + cols
        valid = block * BLOCK_K + cols < D
        a = ct.where(valid, ct.gather(anchor, offsets), 0.0).astype(ct.float32)
        p = ct.where(valid, ct.gather(positive, offsets), 0.0).astype(ct.float32)
        n = ct.where(valid, ct.gather(negative, offsets), 0.0).astype(ct.float32)
        pos = a - p + eps
        neg = a - n + eps
        positive_total += ct.sum(pos * pos, axis=0)
        negative_total += ct.sum(neg * neg, axis=0)
    loss = ct.maximum(ct.sqrt(positive_total) - ct.sqrt(negative_total) + margin, 0.0)
    ct.store(rows, index=(row,), tile=loss)


@ct.kernel
def mean_kernel(values, output, B: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    total = 0.0
    for block in range(ct.cdiv(B, BLOCK_SIZE)):
        indices = block * BLOCK_SIZE + offsets
        total += ct.sum(ct.where(indices < B, ct.gather(values, indices), 0.0), axis=0)
    ct.store(output, index=(0,), tile=total / B)


class Model(nn.Module):
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, anchor, positive, negative):
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negative = negative.contiguous()
        B, D = anchor.shape
        rows = torch.empty(B, device=anchor.device, dtype=torch.float32)
        output = torch.empty(1, device=anchor.device, dtype=torch.float32)
        triplet_kernel(None, (anchor.view(-1), positive.view(-1), negative.view(-1), rows, B, D, self.margin, self.eps))
        with cpu.compile_options({"assume_in_bounds": B % 128 == 0}):
            ct.launch(None, (1,), mean_kernel, (rows, output, B, 128))
        return output[0]
