# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _gemm_epilogue(value):
    return value / 2.0


def _make_reduction_post_op(scaling_factor):
    def reduction_post_op(value, **kwargs):
        return value * scaling_factor

    return reduction_post_op


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self._scaling_factor = float(scaling_factor)
        self._reduction_post_op = _make_reduction_post_op(
            self._scaling_factor
        )
        self._weight_packed = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
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
            reduction_post_op=self._reduction_post_op,
            keep_dim=True,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
