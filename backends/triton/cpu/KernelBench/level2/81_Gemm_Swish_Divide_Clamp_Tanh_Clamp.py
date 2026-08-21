# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import tanh


@triton.jit
def _epilogue(x):
    x *= tl.sigmoid(x)
    x /= 2.0
    x = tl.clamp(x, -1.0, 1.0)
    x = tanh(x)
    x = tl.clamp(x, -1.0, 1.0)
    return x


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            post_op=_epilogue,
        )
