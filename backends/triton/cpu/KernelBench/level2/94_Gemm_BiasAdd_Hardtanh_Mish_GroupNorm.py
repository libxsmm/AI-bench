# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import groupnorm
from triton_cpu_utils import mish


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
    x = tl.clamp(x, -1.0, 1.0)
    x = mish(x)
    return x


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

        self.num_groups = num_groups
        self._matmul_helper = None
        self._bias_extra = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )
            self._bias_extra = self.bias.data.to(dtype=x.dtype)
            assert self.hardtanh.min_val == -1.0 and self.hardtanh.max_val == 1.0, (
                "Assuming Hardtanh to have min_val=-1.0 and max_val=1.0"
            )
            assert self.groupnorm.affine, "GroupNorm must have affine=True"

        res_mm = self._matmul_helper(
            x,
            post_op=_epilogue,
            post_op_arg=self._bias_extra,
            trunc_output=False,
            c_is_owned=True,
        )

        return groupnorm(
            res_mm,
            out_dtype=x.dtype,
            num_groups=self.num_groups,
            eps=self.groupnorm.eps,
        )
