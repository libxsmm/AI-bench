# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
from cutile_cpu_utils import tanh
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _epilogue(value):
    value = value * (1.0 / (1.0 + ct.exp(-value)))
    value = value / 2.0
    value = ct.minimum(ct.maximum(value, -1.0), 1.0)
    value = tanh(value)
    return ct.minimum(ct.maximum(value, -1.0), 1.0)


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype).contiguous()

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
