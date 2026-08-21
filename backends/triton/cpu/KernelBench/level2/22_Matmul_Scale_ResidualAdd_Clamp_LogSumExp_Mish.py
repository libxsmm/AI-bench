# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import mish


@triton.jit
def _red_block(block, BLOCK_SIZE_M, BLOCK_SIZE_N):
    return tl.sum(tl.exp(block), axis=1)


@triton.jit
def _red_epi(val, M, N):
    val = tl.log(val)  # second part of logsumexp
    return val * mish(val)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)

        sf_val = tl.constexpr(scale_factor * 2)
        clmin_val = tl.constexpr(clamp_min)
        clmax_val = tl.constexpr(clamp_max)

        @triton.jit
        def _mm_epi(x):
            return tl.clamp(x * sf_val, clmin_val, clmax_val)

        self._mm_epi_fun = _mm_epi

        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.matmul.weight.data.to(dtype=x.dtype),
                self.matmul.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            post_op=self._mm_epi_fun,
            reduce_last_dim=True,
            reduction_block_op=_red_block,
            reduction_post_op=_red_epi,
            keep_dim=True,
        )
