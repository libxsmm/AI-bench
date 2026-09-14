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
def gelu_kernel(x, output, n_elements: ConstInt, BLOCK_SIZE: ConstInt):
    tile_index = ct.bid(0)
    values = ct.load(x, (tile_index,), (BLOCK_SIZE,), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    absolute_values = ct.abs(values)
    t = 1.0 / (1.0 + 0.3275911 * absolute_values)
    polynomial = t * (
        1.061405429
        + t * (-1.453152027 + t * (1.421413741 + t * (-0.284496736 + t * 0.254829592)))
    )
    erf_values = ct.where(
        values < 0.0,
        -(1.0 - polynomial * ct.exp(-absolute_values * absolute_values)),
        1.0 - polynomial * ct.exp(-absolute_values * absolute_values),
    )
    result = 0.5 * values * (1.0 + erf_values)
    ct.store(output, (tile_index,), ct.astype(result, x.dtype))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor):
        x_flat = x.contiguous().view(-1)
        output = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        gelu_kernel(None, (x_flat, output, n_elements))
        return output.view(x.shape)
