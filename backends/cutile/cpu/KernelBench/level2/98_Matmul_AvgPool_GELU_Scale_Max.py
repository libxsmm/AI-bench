# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import gelu
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _reduction_init(dtype, BLOCK_SIZE_M, **kwargs):
    return ct.full((BLOCK_SIZE_M,), float("-inf"), dtype=dtype)


def _reduction_combine(lhs, rhs, **kwargs):
    return ct.maximum(lhs, rhs)


def _make_reduction_block(pool_kernel_size, scale_factor):
    pool_kernel_size = int(pool_kernel_size)
    scale_factor = float(scale_factor)

    def reduction_block(value, BLOCK_SIZE_M, BLOCK_SIZE_N, **kwargs):
        value = ct.reshape(
            value,
            (BLOCK_SIZE_M, BLOCK_SIZE_N // pool_kernel_size, pool_kernel_size),
        )
        value = ct.sum(value, axis=2) / pool_kernel_size
        value = gelu(value) * scale_factor
        return ct.max(value, axis=1)

    return reduction_block


class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self._weight_packed = None
        self._bias = None
        self._reduction_block = _make_reduction_block(
            pool_kernel_size, scale_factor
        )

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
            reduction_init_val=_reduction_init,
            reduction_block_op=self._reduction_block,
            reduction_combine_op=_reduction_combine,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
