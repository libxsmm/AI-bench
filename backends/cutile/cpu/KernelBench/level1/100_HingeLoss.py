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
    key=["D"],
    grid=lambda meta: (meta["B"],),
    options=lambda meta: {"assume_in_bounds": meta["D"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def hinge_row_kernel(pred, target, rows, B: ConstInt, D: ConstInt, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    pred_mem = pred.get_raw_memory()
    target_mem = target.get_raw_memory()
    total = 0.0
    for col_start in range(0, D, BLOCK_SIZE):
        offsets = row * D + col_start + cols
        target_offsets = col_start + cols
        valid = col_start + cols < D
        predictions = pred_mem.load_offset(offsets, mask=valid, padding_value=0.0).astype(ct.float32)
        targets = target_mem.load_offset(target_offsets, mask=valid, padding_value=0.0).astype(ct.float32)
        total += ct.sum(ct.maximum(1.0 - predictions * targets, 0.0), axis=0)
    ct.store(rows, index=(row,), tile=total)


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        B, D = predictions.shape
        rows = torch.empty(B, device=predictions.device, dtype=torch.float32)
        hinge_row_kernel(None, (predictions.view(-1), targets.view(-1), rows, B, D))
        
        return rows.sum() / (B * D)
