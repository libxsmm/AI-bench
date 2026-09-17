# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cuda.tile._backend import cpu
from cutile_cpu_utils import gelu
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _epilogue(value, block_n, post_op_arg_ptr, **kwargs):
    neg_bias = ct.load(
        post_op_arg_ptr,
        index=(block_n,),
        shape=(32,),
    ).astype(value.dtype)
    return value - neg_bias[None, :]


def _reduction_post_op(value, N, **kwargs):
    return gelu(value / N)


@ct.kernel
def _residual_add(
    column_values,
    original_matrix,
    output_matrix,
    K: ConstInt,
    BLOCK_SIZE_K: ConstInt,
):
    row = ct.bid(0)
    row_mean = ct.load(column_values, index=(row,), shape=())
    for block in range(ct.cdiv(K, BLOCK_SIZE_K)):
        values = ct.load(
            original_matrix,
            index=(row, block),
            shape=(1, BLOCK_SIZE_K),
        )
        values = ct.reshape(values, (BLOCK_SIZE_K,))
        values = values + row_mean
        ct.store(
            output_matrix,
            index=(row, block),
            tile=ct.reshape(values.astype(output_matrix.dtype), (1, BLOCK_SIZE_K)),
        )


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._weight_packed = None
        self._bias = None
        self._neg_bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = (
                None
                if self.gemm.bias is None
                else self.gemm.bias.data.to(dtype=x.dtype).contiguous()
            )
            self._neg_bias = self.subtract.data.to(dtype=x.dtype).contiguous()

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            post_op_arg=self._neg_bias,
            reduce_last_dim=True,
            reduction_post_op=_reduction_post_op,
            keep_dim=True,
            trunc_output=False,
            b_is_prepacked=True,
            c_is_owned=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        output = torch.empty_like(x)
        assert x.shape[1] % 256 == 0, "K must be divisible by BLOCK_SIZE_K"
        with cpu.compile_options({"assume_in_bounds": True}):
            ct.launch(
                None,
                (x.shape[0],),
                _residual_add,
                (res_mm.view(-1), x, output, x.shape[1], 256),
            )
        return output
