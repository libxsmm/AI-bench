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
    key=["OL", "KERNEL_SIZE"],
    grid=lambda meta: (meta["x"].shape[0] // meta["L"], ct.cdiv(meta["OL"], meta["BLOCK_SIZE"])),
    options=lambda meta: {"assume_in_bounds": meta["OL"] % meta["BLOCK_SIZE"] == 0 and meta["PADDING"] == 0},
)
@ct.kernel
def avg_pool1d_kernel(x, output, L: ConstInt, OL: ConstInt, C: ConstInt, KERNEL_SIZE: ConstInt, STRIDE: ConstInt, PADDING: ConstInt, BLOCK_SIZE: ConstInt):
    pid_bc = ct.bid(0)
    pid_o = ct.bid(1)
    offs = pid_o * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=torch.int32)
    valid_out = offs < OL
    base = pid_bc * L
    channel_x = x.slice(axis=0, start=base, stop=base + L)
    x_view = channel_x.tiled_view(
        BLOCK_SIZE,
        padding_mode=ct.PaddingMode.ZERO,
        traversal_steps=1,
    )
    acc = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    for k in range(KERNEL_SIZE):
        indices = offs * STRIDE - PADDING + k
        valid = valid_out & (indices >= 0) & (indices < L)
        if STRIDE == 1:
            start = pid_o * BLOCK_SIZE - PADDING + k
            values = x_view.load(start).astype(ct.float32)
            values = ct.where(valid, values, 0.0)
        else:
            values = ct.where(valid, ct.gather(x, base + indices), 0.0).astype(ct.float32)
        acc += values
    ct.scatter(output, pid_bc * OL + offs, ct.astype(acc / KERNEL_SIZE, output.dtype))


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()
        OL = (L + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((B, C, OL), device=x.device, dtype=torch.bfloat16)
        avg_pool1d_kernel(None, (x.view(-1), output.view(-1), L, OL, C, self.kernel_size, self.stride, self.padding))
        return output
