# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]


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
            bias=self._bias,
            post_op=self._epilogue_fun,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
