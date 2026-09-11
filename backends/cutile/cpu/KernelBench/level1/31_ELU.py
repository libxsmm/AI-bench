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
def elu_kernel(x, output, alpha, n_elements: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.bid(0) * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    values = ct.gather(x, offsets)
    exp_x = ct.exp2(values * 1.4426950408889634)
    result = ct.where(values > 0.0, values, alpha * (exp_x - 1.0))
    ct.scatter(output, offsets, result)


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super(Model, self).__init__()
        try:
            self.alpha = float(alpha)
        except (ValueError, TypeError):
            self.alpha = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.contiguous().view(-1)
        output = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        elu_kernel(None, (x_flat, output, self.alpha, n_elements))
        return output.view_as(x)
