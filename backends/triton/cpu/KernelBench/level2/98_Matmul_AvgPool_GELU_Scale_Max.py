# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


@triton.jit
def _red_init(dtype, BLOCK_SIZE_M, **kwargs):
    return tl.full([BLOCK_SIZE_M], float("-inf"), dtype=dtype)


@triton.jit
def _red_combine(a, b, **kwargs):
    return tl.maximum(a, b)


class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

        kern_sz = tl.constexpr(pool_kernel_size)
        sf_val = tl.constexpr(scale_factor)

        @triton.jit
        def _red_block(block, BLOCK_SIZE_M, BLOCK_SIZE_N, **kwargs):
            block = block.reshape([BLOCK_SIZE_M, BLOCK_SIZE_N // kern_sz, kern_sz])
            xmean = tl.sum(block, axis=2) / kern_sz
            xmean = gelu(xmean) * sf_val
            return tl.max(xmean, axis=1)

        self._weight_packed = None
        self._bias = None
        self._red_block_fun = _red_block

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype)

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            b_is_prepacked=True,
            reduce_last_dim=True,
            reduction_init_val=_red_init,
            reduction_block_op=self._red_block_fun,
            reduction_combine_op=_red_combine,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
