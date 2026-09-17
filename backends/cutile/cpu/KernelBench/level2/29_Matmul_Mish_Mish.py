# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _softplus(value):
    return ct.where(
        value > 20.0,
        value,
        ct.log(1.0 + ct.exp(value)),
    )


def _mish(value):
    return value * ct.tanh(_softplus(value))


def _mish_mish(value):
    return _mish(_mish(value))


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.linear.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.linear.bias.data.to(dtype=x.dtype).contiguous()

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_mish_mish,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
