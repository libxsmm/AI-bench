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
    configs=[ct.tune.Config({"BLOCK_W": 16})],
    key=["OW"],
    grid=lambda meta: (meta["x"].shape[0] // (meta["H"] * meta["W"]), meta["OH"], ct.cdiv(meta["OW"], meta["BLOCK_W"])),
    options=lambda meta: {"assume_in_bounds": meta["OW"] % meta["BLOCK_W"] == 0},
)
@ct.kernel
def avg_pool2d_kernel(x, output, H: ConstInt, W: ConstInt, OH: ConstInt, OW: ConstInt, KERNEL_SIZE: ConstInt, STRIDE: ConstInt, BLOCK_W: ConstInt):
    pid_nc = ct.bid(0)
    pid_oh = ct.bid(1)
    pid_ow = ct.bid(2)
    cols = pid_ow * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32)
    base = pid_nc * H * W
    acc = ct.full((BLOCK_W,), 0.0, dtype=ct.float32)
    for kh in range(KERNEL_SIZE):
        ih = pid_oh * STRIDE + kh
        for kw in range(KERNEL_SIZE):
            iw = cols * STRIDE + kw
            acc += ct.where(cols < OW, ct.gather(x, base + ih * W + iw), 0.0).astype(ct.float32)
    ct.scatter(output, pid_nc * OH * OW + pid_oh * OW + cols, ct.astype(acc / (KERNEL_SIZE * KERNEL_SIZE), output.dtype))


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.contiguous()
        OH = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        OW = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        avg_pool2d_kernel(None, (x.view(-1), output.view(-1), H, W, OH, OW, self.kernel_size, self.stride))
        return output
