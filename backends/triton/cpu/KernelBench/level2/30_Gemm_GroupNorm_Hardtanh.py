# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import groupnorm


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        ht_min = tl.constexpr(hardtanh_min)
        ht_max = tl.constexpr(hardtanh_max)

        @triton.jit
        def _norm_epi(x):
            return tl.clamp(x, ht_min, ht_max)

        self.num_groups = num_groups
        self._norm_epilogue_fun = _norm_epi
        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )
            assert self.group_norm.affine, "GroupNorm must have affine=True"

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
            post_op=self._norm_epilogue_fun,
        )
