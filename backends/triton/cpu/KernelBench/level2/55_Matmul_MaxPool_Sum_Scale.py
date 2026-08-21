# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper


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

        self._matmul_helper = None
        self._red_block_fun = _red_block
        self._red_epi_fun = _red_epi

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.matmul.weight.data.to(dtype=x.dtype),
                self.matmul.bias.data.to(dtype=x.dtype),
            )

        return self._matmul_helper(
            x,
            reduce_last_dim=True,
            reduction_block_op=self._red_block_fun,
            reduction_post_op=self._red_epi_fun,
        )
