# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


@triton.jit
def _gemm_epilogue(x):
    x /= 2.0
    return x


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

        sf_val = tl.constexpr(scaling_factor)

        @triton.jit
        def _red_epilogue(x, M, N):
            x *= sf_val
            return x

        self._red_epilogue_fun = _red_epilogue
        self._weight_packed = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )

        return sfc_matmul(
            x,
            self._weight_packed,
            post_op=_gemm_epilogue,
            reduce_last_dim=True,
            reduction_post_op=self._red_epilogue_fun,
            keep_dim=True,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
