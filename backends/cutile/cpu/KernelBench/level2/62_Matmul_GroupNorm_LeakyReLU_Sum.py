# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import groupnorm
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _make_norm_epilogue(negative_slope):
    negative_slope = float(negative_slope)

    def norm_epilogue(value):
        value = ct.where(value >= 0, value, value * negative_slope)
        return value * 2.0

    return norm_epilogue


class Model(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=hidden_size,
            eps=eps,
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.num_groups = num_groups
        self._norm_epilogue_fun = _make_norm_epilogue(negative_slope)
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.fc.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.fc.bias.data.to(dtype=x.dtype).contiguous()
            assert self.gn.affine, "GroupNorm must have affine=True"

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
            eps=self.gn.eps,
            post_op=self._norm_epilogue_fun,
        )
