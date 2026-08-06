# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)

        kern_sz = tl.constexpr(kernel_size)
        sf_val = tl.constexpr(scale_factor)

        @triton.jit
        def _red(val, x, BLOCK_SIZE_N: tl.constexpr):
            x = x.reshape([BLOCK_SIZE_N // kern_sz, kern_sz])
            xm = tl.max(x, axis=1)
            return val + tl.sum(xm, axis=0)

        @triton.jit
        def _red_epi(val, nelem):
            val *= sf_val
            return val

        self._weight_packed = None
        self._bias = None
        self._red_fun = _red
        self._red_epi_fun = _red_epi

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
            bias=self._bias,
            b_is_prepacked=True,
            trunc_output=False,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        return reduce_last_dim(
            res_mm,
            out_dtype=x.dtype,
            reduction=self._red_fun,
            post_op=self._red_epi_fun,
        )
