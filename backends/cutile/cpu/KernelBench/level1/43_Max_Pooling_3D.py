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
    grid=lambda meta: (ct.cdiv(meta["OW"], meta["BLOCK_W"]), meta["OD"] * meta["OH"], meta["B"] * meta["C"]),
    options=lambda meta: {"assume_in_bounds": meta["OW"] % meta["BLOCK_W"] == 0 and meta["PADDING"] == 0 and meta["DILATION"] == 1},
)
@ct.kernel
def maxpool3d_kernel(x, output, B: ConstInt, C: ConstInt, D: ConstInt, H: ConstInt, W: ConstInt, OD: ConstInt, OH: ConstInt, OW: ConstInt, KERNEL_SIZE: ConstInt, STRIDE: ConstInt, PADDING: ConstInt, DILATION: ConstInt, BLOCK_W: ConstInt):
    pid_ow = ct.bid(0)
    pid_dh = ct.bid(1)
    pid_bc = ct.bid(2)
    batch = pid_bc // C
    channel = pid_bc % C
    od = pid_dh // OH
    oh = pid_dh % OH
    cols = pid_ow * BLOCK_W + ct.arange(BLOCK_W, dtype=torch.int32)
    base = batch * C * D * H * W + channel * D * H * W
    x_view = x.tiled_view(
        BLOCK_W,
        padding_mode=ct.PaddingMode.NEG_INF,
        traversal_steps=1,
    )
    max_value = ct.full((BLOCK_W,), float("-inf"), dtype=ct.float32)
    for kd in range(KERNEL_SIZE):
        d = od * STRIDE - PADDING + kd * DILATION
        if d >= 0 and d < D:
            for kh in range(KERNEL_SIZE):
                h = oh * STRIDE - PADDING + kh * DILATION
                if h >= 0 and h < H:
                    for kw in range(KERNEL_SIZE):
                        w = cols * STRIDE - PADDING + kw * DILATION
                        valid = (cols < OW) & (w >= 0) & (w < W)
                        if STRIDE == 1:
                            start = base + d * H * W + h * W + pid_ow * BLOCK_W - PADDING + kw * DILATION
                            values = x_view.load(start).astype(ct.float32)
                            values = ct.where(valid, values, float("-inf"))
                        else:
                            values = ct.where(valid, ct.gather(x, base + d * H * W + h * W + w), float("-inf")).astype(ct.float32)
                        max_value = ct.maximum(max_value, values)
    output_base = batch * C * OD * OH * OW + channel * OD * OH * OW + od * OH * OW + oh * OW
    ct.scatter(output, output_base + cols, ct.astype(max_value, output.dtype))


def maxpool3d(x, kernel_size, stride, padding, dilation):
    B, C, D, H, W = x.shape
    OD = (D + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output = torch.empty((B, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    maxpool3d_kernel(None, (x.view(-1), output.view(-1), B, C, D, H, W, OD, OH, OW, kernel_size, stride, padding, dilation))
    return output


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)
