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
    configs=[ct.tune.Config({"BLOCK_S": 32})],
    key=["S", "F"],
    grid=lambda meta: (meta["B"] * ct.cdiv(meta["S"], meta["BLOCK_S"]),),
    options=lambda meta: {"assume_in_bounds": meta["S"] % meta["BLOCK_S"] == 0},
)
@ct.kernel
def rms_norm_kernel(x, output, B: ConstInt, F: ConstInt, S: ConstInt, eps, BLOCK_S: ConstInt):
    pid = ct.bid(0)
    blocks = ct.cdiv(S, BLOCK_S)
    batch = pid // blocks
    block = pid % blocks
    s = block * BLOCK_S + ct.arange(BLOCK_S, dtype=torch.int32)
    valid = s < S
    sum_sq = ct.full((BLOCK_S,), 0.0, dtype=ct.float32)
    for f in range(F):
        offsets = batch * F * S + f * S + s
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        sum_sq += values * values
    rms = ct.sqrt(sum_sq / F + eps)
    for f in range(F):
        offsets = batch * F * S + f * S + s
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        ct.scatter(output, offsets, ct.astype(values / rms, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()
        B, F = x.shape[:2]
        S = 1
        for size in x.shape[2:]:
            S *= size
        x_flat = x.view(-1)
        output = torch.empty_like(x_flat)
        rms_norm_kernel(None, (x_flat, output, B, F, S, self.eps))
        return output.view_as(x)
