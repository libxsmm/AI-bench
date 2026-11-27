# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton

from triton_cpu_utils import get_matmul_block_sizes
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import region_matmul


class Model(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        residual_add_factor = triton.language.constexpr(1 + scaling_factor)

        @triton.jit
        def _epilogue(x):
            return x * residual_add_factor

        self._epilogue_fun = _epilogue
        self._weight_packed = None
        self._bias = None
        self._block_sizes = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._block_sizes = get_matmul_block_sizes(x.dtype)
            _, block_size_n, block_size_k = self._block_sizes
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.linear.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=block_size_n,
                BLOCK_SIZE_K=block_size_k,
            )
            self._bias = self.linear.bias.data.to(dtype=x.dtype).contiguous()

        block_size_m, block_size_n, block_size_k = self._block_sizes
        return region_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=self._epilogue_fun,
            b_is_prepacked=True,
            region_grid=(64, 1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )
