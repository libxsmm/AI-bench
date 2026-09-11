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
    configs=[ct.tune.Config({"BLOCK_H": 1, "BLOCK_W": 32})],
    key=["OH", "OW"],
    grid=lambda meta: (meta["x"].shape[0] // (meta["H"] * meta["W"]), ct.cdiv(meta["OH"], meta["BLOCK_H"]), ct.cdiv(meta["OW"], meta["BLOCK_W"])),
    options=lambda meta: {"assume_in_bounds": meta["OW"] % meta["BLOCK_W"] == 0 and meta["PADDING"] == 0 and meta["DILATION"] == 1},
)
@ct.kernel
def maxpool2d_kernel(x, output, H: ConstInt, W: ConstInt, OH: ConstInt, OW: ConstInt, KERNEL_SIZE: ConstInt, STRIDE: ConstInt, PADDING: ConstInt, DILATION: ConstInt, BLOCK_H: ConstInt, BLOCK_W: ConstInt):
    pid_bc = ct.bid(0)
    pid_oh = ct.bid(1)
    pid_ow = ct.bid(2)
    cols = pid_ow * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32)
    rows = pid_oh * BLOCK_H
    valid_out = cols < OW
    base = pid_bc * H * W
    x_view = x.tiled_view(
        BLOCK_W,
        padding_mode=ct.PaddingMode.NEG_INF,
        traversal_steps=1,
    )
    max_value = ct.full((BLOCK_W,), float("-inf"), dtype=ct.float32)
    for kh in range(KERNEL_SIZE):
        ih = rows * STRIDE - PADDING + kh * DILATION
        if (ih >= 0) & (ih < H):
            for kw in range(KERNEL_SIZE):
                iw = cols * STRIDE - PADDING + kw * DILATION
                valid = valid_out & (iw >= 0) & (iw < W)
                start = base + ih * W + pid_ow * BLOCK_W - PADDING + kw * DILATION
                values = x_view.load(start).astype(ct.float32)
                max_value = ct.maximum(max_value, ct.where(valid, values, float("-inf")))
    ct.scatter(output, pid_bc * OH * OW + rows * OW + cols, ct.astype(max_value, output.dtype))


def maxpool2d(x, kernel_size, stride, padding, dilation):
    B, C, H, W = x.shape
    OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    x = x.contiguous()
    output = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    maxpool2d_kernel(None, (x.view(-1), output.view(-1), H, W, OH, OW, kernel_size, stride, padding, dilation))
    return output


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)
