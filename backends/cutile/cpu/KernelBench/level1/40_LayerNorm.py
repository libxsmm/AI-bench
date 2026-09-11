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
def _layer_norm_kernel(x, output, weight, bias, M: ConstInt, N: ConstInt, eps, BLOCK_SIZE: ConstInt):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    base = row * N
    total = 0.0
    total_sq = 0.0
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = base + block * BLOCK_SIZE + cols
        valid = block * BLOCK_SIZE + cols < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        total += ct.sum(values, axis=0)
        total_sq += ct.sum(values * values, axis=0)
    mean = total / N
    inv_std = 1.0 / ct.sqrt(total_sq / N - mean * mean + eps)
    for block in range(ct.cdiv(N, BLOCK_SIZE)):
        offsets = base + block * BLOCK_SIZE + cols
        param_offsets = block * BLOCK_SIZE + cols
        valid = param_offsets < N
        values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
        w = ct.where(valid, ct.gather(weight, param_offsets), 1.0).astype(ct.float32)
        b = ct.where(valid, ct.gather(bias, param_offsets), 0.0).astype(ct.float32)
        result = (values - mean) * inv_std * w + b
        ct.scatter(output, offsets, ct.astype(result, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._moved = False

    def _move_params(self, device):
        self.w_flat = self.ln.weight.data.to(device, dtype=torch.bfloat16).contiguous().flatten()
        self.b_flat = self.ln.bias.data.to(device, dtype=torch.bfloat16).contiguous().flatten()
        self._eps = self.ln.eps
        self._norm_n = 1
        for size in self.ln.normalized_shape:
            self._norm_n *= size
        self._moved = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._moved:
            self._move_params(x.device)
        x = x.contiguous()
        original_shape = x.shape
        N = self._norm_n
        M = x.numel() // N
        x_flat = x.view(-1)
        output = torch.empty_like(x_flat)
        BLOCK_SIZE = 32
        with cpu.compile_options({"assume_in_bounds": N % BLOCK_SIZE == 0}):
            ct.launch(None, (M,), _layer_norm_kernel, (x_flat, output, self.w_flat, self.b_flat, M, N, self._eps, BLOCK_SIZE))
        return output.view(original_shape)
