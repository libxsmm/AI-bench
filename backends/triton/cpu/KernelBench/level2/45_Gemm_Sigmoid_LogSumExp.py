# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul


@triton.jit
def _mm1_epi(x):
    return tl.sigmoid(x)


@triton.jit
def _red(val, x, **kwargs):
    x = tl.exp(x)
    return val + tl.sum(x, axis=0)


@triton.jit
def _red_epi(val, **kwargs):
    return tl.log(val)  # second part of logsumexp


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
        if self._weight1_packed is None:
            # AMX-optimized block size
            self._weight1_packed = pack_weights_for_sfc_matmul(
                self.linear1.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias1 = self.linear1.bias.data.to(dtype=x.dtype)
            self._weight2_packed = pack_weights_for_sfc_matmul(
                self.linear2.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias2 = self.linear2.bias.data.to(dtype=x.dtype)

        res_mm1 = sfc_matmul(
            x,
            self._weight1_packed,
            self._bias1,
            post_op=_mm1_epi,
            trunc_output=True,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        res_mm2 = sfc_matmul(
            res_mm1,
            self._weight2_packed,
            self._bias2,
            trunc_output=False,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, res_mm1.shape[1] // 4096)),
        )

        res_red = reduce_last_dim(
            res_mm2,
            out_dtype=x.dtype,
            reduction=_red,
            post_op=_red_epi,
        )

        return res_red
