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
def _partial_sum_sq_kernel(x, partial, N: ConstInt, BLOCK_SIZE: ConstInt):
    pid = ct.bid(0)
    offsets = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    valid = offsets < N
    values = ct.where(valid, ct.gather(x, offsets), 0.0).astype(ct.float32)
    ct.store(partial, index=(pid,), tile=ct.sum(values * values, axis=0))


@ct.kernel
def _reduce_kernel(partial, inverse_norm, num_partial: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.arange(BLOCK_SIZE, dtype=torch.int32)
    total = 0.0
    for block in range(ct.cdiv(num_partial, BLOCK_SIZE)):
        indices = block * BLOCK_SIZE + offsets
        total += ct.sum(ct.where(indices < num_partial, ct.gather(partial, indices), 0.0), axis=0)
    ct.store(inverse_norm, index=(0,), tile=1.0 / ct.sqrt(total))


@ct.autotune(
    configs=[ct.tune.Config({"BLOCK_SIZE": 32})],
    key=["N"],
    grid=lambda meta: (ct.cdiv(meta["N"], meta["BLOCK_SIZE"]),),
    options=lambda meta: {"assume_in_bounds": meta["N"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def _normalize_kernel(x, output, inverse_norm, N: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.bid(0) * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    values = ct.where(offsets < N, ct.gather(x, offsets), 0.0).astype(ct.float32)
    scale = ct.load(inverse_norm, index=(0,), shape=())
    ct.scatter(output, offsets, ct.astype(values * scale, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.contiguous().view(-1)
        N = x_flat.numel()
        partial_count = (N + 8191) // 8192
        partial = torch.empty(partial_count, device=x.device, dtype=torch.float32)
        inverse_norm = torch.empty(1, device=x.device, dtype=torch.float32)
        output = torch.empty_like(x_flat)
        with cpu.compile_options({"assume_in_bounds": N % 8192 == 0}):
            ct.launch(None, (partial_count,), _partial_sum_sq_kernel, (x_flat, partial, N, 8192))
        with cpu.compile_options({"assume_in_bounds": partial_count % 128 == 0}):
            ct.launch(None, (1,), _reduce_kernel, (partial, inverse_norm, partial_count, 128))
        _normalize_kernel(None, (x_flat, output, inverse_norm, N))
        return output.view(original_shape)
