# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper


@triton.jit
def _gemm_epilogue(x):
    x /= 2.0
    return x


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

        sf_val = tl.constexpr(scaling_factor)

        @triton.jit
        def _red_epilogue(x, M, N):
            x *= sf_val
            return x

        self._red_epilogue_fun = _red_epilogue
        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.weight.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            post_op=_gemm_epilogue,
            reduce_last_dim=True,
            reduction_post_op=self._red_epilogue_fun,
            keep_dim=True,
        )
