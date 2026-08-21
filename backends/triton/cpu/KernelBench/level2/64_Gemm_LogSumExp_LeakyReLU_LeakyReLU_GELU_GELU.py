# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import gelu


@triton.jit
def _red_block(block, BLOCK_SIZE_M, BLOCK_SIZE_N):
    return tl.sum(tl.exp(block), axis=1)


@triton.jit
def _red_epi(val, M, N):
    NEG_SLOPE: tl.constexpr = 0.01
    val = tl.log(val)  # second part of logsumexp
    val = tl.where(val >= 0, val, val * NEG_SLOPE)  # leaky relu
    val = tl.where(val >= 0, val, val * NEG_SLOPE)
    val = gelu(val)
    val = gelu(val)
    return val


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.linear.weight.data.to(dtype=x.dtype),
                self.linear.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            reduce_last_dim=True,
            reduction_block_op=_red_block,
            reduction_post_op=_red_epi,
            keep_dim=True,
        )
