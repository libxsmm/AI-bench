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
    configs=[ct.tune.Config({"BLOCK_W": 32})],
    key=["OW"],
    grid=lambda meta: (ct.cdiv(meta["OW"], meta["BLOCK_W"]), meta["B"] * meta["C"] * meta["OD"] * meta["OH"]),
    options=lambda meta: {"assume_in_bounds": meta["OW"] % meta["BLOCK_W"] == 0 and meta["PADDING"] == 0},
)
@ct.kernel
def avg_pool3d_kernel(x, output, B: ConstInt, C: ConstInt, D: ConstInt, H: ConstInt, W: ConstInt, OD: ConstInt, OH: ConstInt, OW: ConstInt, KERNEL_SIZE: ConstInt, STRIDE: ConstInt, PADDING: ConstInt, BLOCK_W: ConstInt):
    pid_ow = ct.bid(0)
    pid_ncdoh = ct.bid(1)
    oh = pid_ncdoh % OH
    tmp = pid_ncdoh // OH
    od = tmp % OD
    tmp = tmp // OD
    channel = tmp % C
    batch = tmp // C
    cols = pid_ow * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32)
    base = batch * C * D * H * W + channel * D * H * W
    acc = ct.full((BLOCK_W,), 0.0, dtype=ct.float32)
    for kd in range(KERNEL_SIZE):
        d = od * STRIDE + kd - PADDING
        if d >= 0 and d < D:
            for kh in range(KERNEL_SIZE):
                h = oh * STRIDE + kh - PADDING
                if h >= 0 and h < H:
                    for kw in range(KERNEL_SIZE):
                        w = cols * STRIDE + kw - PADDING
                        valid = (cols < OW) & (w >= 0) & (w < W)
                        acc += ct.where(valid, ct.gather(x, base + d * H * W + h * W + w), 0.0).astype(ct.float32)
    out_base = batch * C * OD * OH * OW + channel * OD * OH * OW + od * OH * OW + oh * OW
    ct.scatter(output, out_base + cols, ct.astype(acc / (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE), ct.bfloat16))


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        OD = (D + 2 * self.padding - self.kernel_size) // self.stride + 1
        OH = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        OW = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((B, C, OD, OH, OW), device=x.device, dtype=x.dtype)
        avg_pool3d_kernel(None, (x.view(-1), output.view(-1), B, C, D, H, W, OD, OH, OW, self.kernel_size, self.stride, self.padding))
        return output
