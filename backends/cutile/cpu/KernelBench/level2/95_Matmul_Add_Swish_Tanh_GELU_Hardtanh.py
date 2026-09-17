# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import gelu
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
from cutile_cpu_utils import tanh
import torch
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _epilogue(value, block_n, post_op_arg_ptr, **kwargs):
    add_value = ct.load(
        post_op_arg_ptr,
        index=(block_n,),
        shape=(32,),
    ).astype(value.dtype)
    value = value + add_value[None, :]
    value = value * (1.0 / (1.0 + ct.exp(-value)))
    value = tanh(value)
    value = gelu(value)
    return ct.minimum(ct.maximum(value, -1.0), 1.0)


class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self._weight_packed = None
        self._bias = None
        self._add_value = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype).contiguous()
            self._add_value = self.add_value.data.to(dtype=x.dtype).contiguous()

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            post_op_arg=self._add_value,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
