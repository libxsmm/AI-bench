# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


@triton.jit
def _red_block(block, **kwargs):
    return tl.sum(tl.exp(block), axis=1)


@triton.jit
def _red_epi(val, **kwargs):
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
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.linear.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.linear.bias.data.to(dtype=x.dtype)

        return sfc_matmul(
            x,
            self._weight_packed,
            self._bias,
            reduce_last_dim=True,
            reduction_block_op=_red_block,
            reduction_post_op=_red_epi,
            keep_dim=True,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
