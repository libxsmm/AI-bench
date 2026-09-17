# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import groupnorm
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _norm_epilogue(
    value,
    g,
    c,
    post_op_arg_ptr,
    C,
    group_size,
    BLOCK_SIZE_C,
    **kwargs,
):
    weight = ct.load(
        post_op_arg_ptr,
        index=(g * (group_size // BLOCK_SIZE_C) + c // BLOCK_SIZE_C,),
        shape=(BLOCK_SIZE_C,),
    ).astype(value.dtype)
    value = value * (1.0 / (1.0 + ct.exp(-value)))
    value = value * weight
    return value * (1.0 / (1.0 + ct.exp(-value)))


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
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype).contiguous()
            self._multiply_weight = self.multiply_weight.data.to(dtype=x.dtype).contiguous()
            assert self.group_norm.affine, "GroupNorm must have affine=True"

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            trunc_output=False,
            b_is_prepacked=True,
            c_is_owned=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
        return groupnorm(
            res_mm,
            out_dtype=x.dtype,
            num_groups=self.num_groups,
            eps=self.group_norm.eps,
            post_op=_norm_epilogue,
            post_op_arg=self._multiply_weight,
        )
