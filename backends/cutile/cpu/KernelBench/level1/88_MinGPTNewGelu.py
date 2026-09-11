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
    key=["n_elements"],
    grid=lambda meta: (ct.cdiv(meta["n_elements"], meta["BLOCK_SIZE"]),),
    options=lambda meta: {"assume_in_bounds": meta["n_elements"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def gelu_kernel(x, output, n_elements: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.bid(0) * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    values = ct.gather(x, offsets).astype(ct.float32)
    inner = values + 0.044715 * values * values * values
    result = 0.5 * values * (1.0 + ct.tanh(0.7978845608028654 * inner))
    ct.scatter(output, offsets, ct.astype(result, x.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, x):
        x_flat = x.contiguous().view(-1)
        output = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        gelu_kernel(None, (x_flat, output, n_elements))
        return output.view(x.shape)
