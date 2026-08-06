# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import mish
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul


@triton.jit
def _red(val, x, BLOCK_SIZE_N: tl.constexpr):
    x = tl.exp(x)
    return val + tl.sum(x, axis=0)


@triton.jit
def _red_epi(val, nelem):
    val = tl.log(val)  # second part of logsumexp
    return val * mish(val)


batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)

        sf_val = tl.constexpr(scale_factor * 2)
        clmin_val = tl.constexpr(clamp_min)
        clmax_val = tl.constexpr(clamp_max)

        @triton.jit
        def _mm_epi(x):
            return tl.clamp(x * sf_val, clmin_val, clmax_val)

        self._mm_epi_fun = _mm_epi

        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype)

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            self._bias,
            post_op=self._mm_epi_fun,
            trunc_output=False,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        res_red = reduce_last_dim(
            res_mm,
            out_dtype=x.dtype,
            reduction=_red,
            post_op=_red_epi,
            keep_dim=True,
        )

        return res_red
