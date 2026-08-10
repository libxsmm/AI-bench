# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul
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
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype)

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
