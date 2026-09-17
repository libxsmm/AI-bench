# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper


class Model(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        mul_val = triton.language.constexpr(multiplier)
        neg_slope_val = triton.language.constexpr(negative_slope)

        @triton.jit
        def _epilogue(x):
            x = x * mul_val
            x = tl.where(x >= 0, x, x * neg_slope_val)
            return x

        self._epilogue_fun = _epilogue
        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.linear.weight.data.to(dtype=x.dtype),
                self.linear.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            post_op=self._epilogue_fun,
        )
