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
    key=["OL"],
    grid=lambda meta: (meta["x"].shape[0] // meta["L"], ct.cdiv(meta["OL"], meta["BLOCK_SIZE"])),
    options=lambda meta: {"assume_in_bounds": meta["OL"] % meta["BLOCK_SIZE"] == 0 and meta["PADDING"] == 0 and meta["DILATION"] == 1},
)
@ct.kernel
def maxpool1d_kernel(x, output, L: ConstInt, OL: ConstInt, C: ConstInt, KERNEL_SIZE: ConstInt, STRIDE: ConstInt, PADDING: ConstInt, DILATION: ConstInt, BLOCK_SIZE: ConstInt):
    pid_bc = ct.bid(0)
    pid_o = ct.bid(1)
    b = pid_bc // C
    c = pid_bc % C
    offs = pid_o * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    valid_out = offs < OL
    base = b * C * L + c * L
    x_view = x.tiled_view(
        BLOCK_SIZE,
        padding_mode=ct.PaddingMode.NEG_INF,
        traversal_steps=1,
    )
    max_value = ct.full((BLOCK_SIZE,), float("-inf"), dtype=ct.float32)
    for k in range(KERNEL_SIZE):
        indices = offs * STRIDE + k * DILATION - PADDING
        valid = valid_out & (indices >= 0) & (indices < L)
        if STRIDE == 1:
            start = base + pid_o * BLOCK_SIZE - PADDING + k * DILATION
            values = x_view.load(start).astype(ct.float32)
            values = ct.where(valid, values, float("-inf"))
        else:
            values = ct.where(valid, ct.gather(x, base + indices), float("-inf")).astype(ct.float32)
        max_value = ct.maximum(max_value, values)
    ct.scatter(output, b * C * OL + c * OL + offs, ct.astype(max_value, output.dtype))


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x = x.contiguous()
        OL = (L + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output = torch.empty((B, C, OL), device=x.device, dtype=x.dtype)
        maxpool1d_kernel(None, (x.view(-1), output.view(-1), L, OL, C, self.kernel_size, self.stride, self.padding, self.dilation))
        return output
