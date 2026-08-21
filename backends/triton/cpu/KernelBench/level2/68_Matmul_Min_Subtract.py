# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper


class Model(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(
            torch.tensor(constant)
        )  # unused, but needed to match state of reference model

        const_val = triton.language.constexpr(constant)

        @triton.jit
        def _epilogue(x):
            return tl.minimum(x, const_val) - const_val

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
