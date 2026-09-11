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
    configs=[ct.tune.Config({"BLOCK_SIZE": size}) for size in [32, 64, 128, 256, 512, 1024, 2048, 4096]],
    key=["n_elements"],
    grid=lambda meta: (ct.cdiv(meta["n_elements"], meta["BLOCK_SIZE"]),),
    options=lambda meta: {"assume_in_bounds": meta["n_elements"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def softplus_kernel(x, output, n_elements: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.bid(0) * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    values = ct.gather(x, offsets)
    result = ct.where(values > 20.0, values, ct.log(1.0 + ct.exp(values)))
    ct.scatter(output, offsets, ct.astype(result, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.contiguous().view(-1)
        output = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        softplus_kernel(None, (x_flat, output, n_elements))
        return output.view_as(x)
