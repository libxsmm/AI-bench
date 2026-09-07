# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import groupnorm


@triton.jit
def _norm_epilogue(x, n, g, c, post_op_arg_ptr, N, C, group_size, BLOCK_SIZE_C):
    desc = tl.make_tensor_descriptor(
        post_op_arg_ptr,
        shape=[C],
        strides=[1],
        block_shape=[BLOCK_SIZE_C],
    )
    x *= tl.sigmoid(x)
    x *= desc.load([g * group_size + c])
    x *= tl.sigmoid(x)
    return x


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

        self.num_groups = num_groups
        self._matmul_helper = None
        self._multiply_weight = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )
            assert self.group_norm.affine, "GroupNorm must have affine=True"
            self._multiply_weight = self.multiply_weight.data.to(dtype=x.dtype)

        res_mm = self._matmul_helper(
            x,
            trunc_output=False,
            c_is_owned=True,
        )

        return groupnorm(
            res_mm,
            out_dtype=x.dtype,
            num_groups=self.num_groups,
            eps=self.group_norm.eps,
            post_op=_norm_epilogue,
            post_op_arg=self._multiply_weight,
        )
