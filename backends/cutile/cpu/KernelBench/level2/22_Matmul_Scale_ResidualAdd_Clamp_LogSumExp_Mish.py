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


def _make_matmul_epilogue(scale_factor, clamp_min, clamp_max):
    scale = float(scale_factor) * 2.0
    clamp_min = float(clamp_min)
    clamp_max = float(clamp_max)

    def matmul_epilogue(value):
        value = value * scale
        return ct.minimum(ct.maximum(value, clamp_min), clamp_max)

    return matmul_epilogue


def _reduction_block(value, **kwargs):
    return ct.sum(ct.exp(value), axis=1)


def _softplus(value):
    return ct.where(
        value > 20.0,
        value,
        ct.log(1.0 + ct.exp(value)),
    )


def _mish(value):
    return value * ct.tanh(_softplus(value))


def _reduction_post_op(value, **kwargs):
    value = ct.log(value)
    return value * _mish(value)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self._matmul_epilogue = _make_matmul_epilogue(
            scale_factor, clamp_min, clamp_max
        )
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype).contiguous()

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=self._matmul_epilogue,
            reduce_last_dim=True,
            reduction_block_op=_reduction_block,
            reduction_post_op=_reduction_post_op,
            keep_dim=True,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
