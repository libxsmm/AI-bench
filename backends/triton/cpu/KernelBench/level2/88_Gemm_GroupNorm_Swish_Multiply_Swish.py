# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import groupnorm
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


@triton.jit
def _norm_epilogue(
    x, n, g, c, post_op_arg_ptr, N, C, group_size, BLOCK_SIZE_C: tl.constexpr
):
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


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

        self.num_groups = num_groups
        self._weight_packed = None
        self._bias = None
        self._multiply_weight = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype)
            assert self.group_norm.affine, "GroupNorm must have affine=True"
            self._multiply_weight = self.multiply_weight.data.to(dtype=x.dtype)

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            b_is_prepacked=True,
            trunc_output=False,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        return groupnorm(
            res_mm,
            out_dtype=x.dtype,
            num_groups=self.num_groups,
            eps=self.group_norm.eps,
            post_op=_norm_epilogue,
            post_op_arg=self._multiply_weight,
        )
