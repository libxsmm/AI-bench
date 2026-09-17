# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _make_reduction_block(kernel_size):
    kernel_size = int(kernel_size)

    def reduction_block(value, BLOCK_SIZE_M, BLOCK_SIZE_N, **kwargs):
        value = ct.reshape(
            value,
            (BLOCK_SIZE_M, BLOCK_SIZE_N // kernel_size, kernel_size),
        )
        pooled = ct.max(value, axis=2)
        return ct.sum(pooled, axis=1)

    return reduction_block


def _make_reduction_post_op(scale_factor):
    scale_factor = float(scale_factor)

    def reduction_post_op(value, **kwargs):
        return value * scale_factor

    return reduction_post_op


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self._reduction_block = _make_reduction_block(kernel_size)
        self._reduction_post_op = _make_reduction_post_op(scale_factor)
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype).contiguous()

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            reduce_last_dim=True,
            reduction_block_op=self._reduction_block,
            reduction_post_op=self._reduction_post_op,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
