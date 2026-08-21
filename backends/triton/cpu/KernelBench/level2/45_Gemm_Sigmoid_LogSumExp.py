# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper


@triton.jit
def _mm1_epi(x):
    return tl.sigmoid(x)


@triton.jit
def _red_block(block, BLOCK_SIZE_M, BLOCK_SIZE_N):
    return tl.sum(tl.exp(block), axis=1)


@triton.jit
def _red_epi(val, M, N):
    return tl.log(val)  # second part of logsumexp


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._matmul_helper1 = None
        self._matmul_helper2 = None

    def forward(self, x):
        if self._matmul_helper1 is None:
            self._matmul_helper1 = SFCMatmulHelper(
                self.linear1.weight.data.to(dtype=x.dtype),
                self.linear1.bias.data.to(dtype=x.dtype),
            )
            self._matmul_helper2 = SFCMatmulHelper(
                self.linear2.weight.data.to(dtype=x.dtype),
                self.linear2.bias.data.to(dtype=x.dtype),
            )

        res_mm1 = self._matmul_helper1(
            x,
            post_op=_mm1_epi,
            trunc_output=True,
            c_is_owned=True,
        )

        return self._matmul_helper2(
            res_mm1,
            reduce_last_dim=True,
            reduction_block_op=_red_block,
            reduction_post_op=_red_epi,
        )
