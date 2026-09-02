# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)

        kern_sz = tl.constexpr(kernel_size)
        sf_val = tl.constexpr(scale_factor)

        @triton.jit
        def _red_block(block, BLOCK_SIZE_M, BLOCK_SIZE_N):
            block = block.reshape([BLOCK_SIZE_M, BLOCK_SIZE_N // kern_sz, kern_sz])
            xm = tl.max(block, axis=2)
            return tl.sum(xm, axis=1)

        @triton.jit
        def _red_epi(val, M, N):
            val *= sf_val
            return val

        self._weight_packed = None
        self._bias = None
        self._red_block_fun = _red_block
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

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            reduce_last_dim=True,
            reduction_block_op=self._red_block_fun,
            reduction_post_op=self._red_epi_fun,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
