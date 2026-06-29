# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


from sfc_matmul import pack_weights_for_sfc_matmul
from sfc_matmul import sfc_matmul
import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        sf = triton.language.constexpr(scaling_factor)
        ht_min = triton.language.constexpr(hardtanh_min)
        ht_max = triton.language.constexpr(hardtanh_max)

        @triton.jit
        def _epilogue(x):
            x = x * sf
            x = tl.clamp(x, ht_min, ht_max)
            x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
            return x

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
            self._bias = self.linear.bias.data.to(dtype=x.dtype)

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=self._epilogue_fun,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
