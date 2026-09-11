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
    configs=[ct.tune.Config({"BLOCK_SIZE": size}) for size in [32, 64, 128, 256, 512, 1024, 2048]],
    key=["n_elements"],
    grid=lambda meta: (ct.cdiv(meta["n_elements"], meta["BLOCK_SIZE"]),),
    options=lambda meta: {"assume_in_bounds": meta["n_elements"] % meta["BLOCK_SIZE"] == 0},
)
@ct.kernel
def scalar_mul_kernel(input, output, scalar, n_elements: ConstInt, BLOCK_SIZE: ConstInt):
    offsets = ct.bid(0) * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    result = ct.gather(input, offsets) * scalar
    ct.scatter(output, offsets, ct.astype(result, ct.bfloat16))


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, s) -> torch.Tensor:
        A = A.contiguous()
        A_flat = A.view(-1)
        output = torch.empty_like(A_flat)
        n_elements = A_flat.numel()
        scalar_val = s.item() if isinstance(s, torch.Tensor) else float(s)
        scalar_mul_kernel(None, (A_flat, output, scalar_val, n_elements))
        return output.view_as(A)
