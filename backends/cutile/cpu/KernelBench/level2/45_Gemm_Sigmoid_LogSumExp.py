# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch.nn as nn

ct.set_backend("cpu")


def _sigmoid(value):
    return value / (1.0 + ct.exp(-value))


def _mm1_epilogue(value):
    return _sigmoid(value)


def _reduction_block(value, **kwargs):
    return ct.sum(ct.exp(value), axis=1)


def _reduction_post_op(value, **kwargs):
    return ct.log(value)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._weight1_packed = None
        self._weight2_packed = None
        self._bias1 = None
        self._bias2 = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight1_packed is None or self._weight1_packed.dtype != x.dtype:
            self._weight1_packed = pack_weights_for_sfc_matmul(
                self.linear1.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias1 = self.linear1.bias.data.to(dtype=x.dtype).contiguous()
            self._weight2_packed = pack_weights_for_sfc_matmul(
                self.linear2.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias2 = self.linear2.bias.data.to(dtype=x.dtype).contiguous()

        res_mm1 = sfc_matmul(
            x,
            self._weight1_packed,
            bias=self._bias1,
            post_op=_mm1_epilogue,
            trunc_output=True,
            b_is_prepacked=True,
            c_is_owned=True,
        )

        return sfc_matmul(
            res_mm1,
            self._weight2_packed,
            bias=self._bias2,
            reduce_last_dim=True,
            reduction_block_op=_reduction_block,
            reduction_post_op=_reduction_post_op,
            b_is_prepacked=True,
        )
