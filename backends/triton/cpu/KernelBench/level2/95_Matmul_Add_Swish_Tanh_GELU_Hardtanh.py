# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import gelu
from triton_cpu_utils import tanh


@triton.jit
def _epilogue(x, block_m, block_n, post_op_arg_ptr, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    desc = tl.make_tensor_descriptor(
        base=post_op_arg_ptr,
        shape=(N,),
        strides=(1,),
        block_shape=(BLOCK_SIZE_N,),
    )
    add_val = desc.load([block_n * BLOCK_SIZE_N]).to(x.dtype)
    x += add_val[None, :]
    x *= tl.sigmoid(x)
    x = tanh(x)
    x = gelu(x)
    x = tl.clamp(x, -1.0, 1.0)
    return x


class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

        self._matmul_helper = None
        self._add_value = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.matmul.weight.data.to(dtype=x.dtype),
                self.matmul.bias.data.to(dtype=x.dtype),
            )
            self._add_value = self.add_value.data.to(dtype=x.dtype)

        return self._matmul_helper(
            x,
            post_op=_epilogue,
            post_op_arg=self._add_value,
        )
