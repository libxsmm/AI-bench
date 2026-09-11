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


@ct.kernel
def instance_norm_kernel(x, output, N: ConstInt, eps, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    base = row * N
    sum_acc = 0.0
    sq_acc = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = base + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        sum_acc += ct.sum(values, axis=0)
        sq_acc += ct.sum(values * values, axis=0)
    mean = sum_acc / N
    inv_std = 1.0 / ct.sqrt(sq_acc / N - mean * mean + eps)
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = base + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        ct.scatter(output, offsets, ct.astype((values - mean) * inv_std, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, num_features: int):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.contiguous().view(B * C, N)
        output = torch.empty_like(x_flat)
        BLOCK_SIZE = 32
        with cpu.compile_options({"assume_in_bounds": N % BLOCK_SIZE == 0}):
            ct.launch(None, (B * C,), instance_norm_kernel, (x_flat.view(-1), output.view(-1), N, self.eps, BLOCK_SIZE))
        return output.view(B, C, H, W)
