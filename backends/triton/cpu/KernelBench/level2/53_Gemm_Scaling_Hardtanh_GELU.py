# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import gelu


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        sf = triton.language.constexpr(scaling_factor)
        ht_min = triton.language.constexpr(hardtanh_min)
        ht_max = triton.language.constexpr(hardtanh_max)

        @triton.jit
        def _epilogue(x):
            x = x * sf
            x = tl.clamp(x, ht_min, ht_max)
            x = gelu(x)
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
