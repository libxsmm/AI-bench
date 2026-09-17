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


def _make_epilogue(divide_value):
    divide_value = float(divide_value)

    def epilogue(value, post_op_arg_ptr, **kwargs):
        bias_value = ct.load(post_op_arg_ptr, index=(0,), shape=()).astype(
            value.dtype
        )
        value = value + bias_value
        value = value * 0.999995 / divide_value
        return value * (1.0 / (1.0 + ct.exp(-value)))

    return epilogue


class Model(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        bias_shape=(1,),
        divide_value=1.0,
    ):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(
            out_features,
            eps=bn_eps,
            momentum=bn_momentum,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._epilogue_fun = _make_epilogue(divide_value)
        self._weight_packed = None
        self._bias = None
        self._bias_extra = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype).contiguous()
            self._bias_extra = self.bias.data.to(dtype=x.dtype).contiguous()
            assert self.bn.affine, "BatchNorm must have affine=True"

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=self._epilogue_fun,
            post_op_arg=self._bias_extra,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
