# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


from sfc_matmul import pack_weights_for_sfc_matmul
from sfc_matmul import sfc_matmul
import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(bias_shape))

        @triton.jit
        def _epilogue(x):
            return tl.maximum(x, 0.0)

        self._epilogue_fun = _epilogue
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.linear.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.bias.data.to(dtype=x.dtype)

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=self._epilogue_fun,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
