# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import gelu


@triton.jit
def _red_init(dtype, BLOCK_SIZE_M):
    return tl.full([BLOCK_SIZE_M], float("-inf"), dtype=dtype)


@triton.jit
def _red_combine(a, b):
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
        def _red_block(block, BLOCK_SIZE_M, BLOCK_SIZE_N):
            block = block.reshape([BLOCK_SIZE_M, BLOCK_SIZE_N // kern_sz, kern_sz])
            xmean = tl.sum(block, axis=2) / kern_sz
            xmean = gelu(xmean) * sf_val
            return tl.max(xmean, axis=1)

        self._matmul_helper = None
        self._red_block_fun = _red_block

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.matmul.weight.data.to(dtype=x.dtype),
                self.matmul.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            reduce_last_dim=True,
            reduction_init_val=_red_init,
            reduction_block_op=self._red_block_fun,
            reduction_combine_op=_red_combine,
        )
