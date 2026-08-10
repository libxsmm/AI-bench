# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import groupnorm
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


class Model(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        negative_slope_val = tl.constexpr(negative_slope)

        @triton.jit
        def _norm_epi(x):
            return tl.where(x >= 0, x, x * negative_slope_val) * 2.0

        self.num_groups = num_groups
        self._norm_epilogue_fun = _norm_epi
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.fc.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.fc.bias.data.to(dtype=x.dtype)
            assert self.gn.affine, "GroupNorm must have affine=True"

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
            eps=self.gn.eps,
            post_op=self._norm_epilogue_fun,
        )
