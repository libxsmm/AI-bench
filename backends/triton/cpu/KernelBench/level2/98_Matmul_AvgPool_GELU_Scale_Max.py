# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]


class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

        kern_sz = tl.constexpr(pool_kernel_size)
        sf_val = tl.constexpr(scale_factor)

        @triton.jit
        def _red(val, x, BLOCK_SIZE_N: tl.constexpr):
            x = x.reshape([BLOCK_SIZE_N // kern_sz, kern_sz])
            xmean = tl.sum(x, axis=1) / kern_sz
            xmean = gelu(xmean) * sf_val
            return tl.maximum(val, tl.max(xmean))

        self._weight_packed = None
        self._bias = None
        self._red_fun = _red

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
        )
