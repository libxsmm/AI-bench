# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul
from triton_cpu_utils import tanh


@triton.jit
def _epilogue(x, m, n, ptr, M, N, BS_M, BS_N):
    desc = tl.make_tensor_descriptor(
        base=ptr,
        shape=(N,),
        strides=(1,),
        block_shape=(BS_N,),
    )
    add_val = desc.load([n * BS_N]).to(x.dtype)
    x += add_val[None, :]
    x *= tl.sigmoid(x)
    x = tanh(x)
    x = gelu(x)
    x = tl.clamp(x, -1.0, 1.0)
    return x


batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, add_value_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

        self._weight_packed = None
        self._bias = None
        self._add_value = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype)
            self._add_value = self.add_value.data.to(dtype=x.dtype)

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            post_op_arg=self._add_value,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
