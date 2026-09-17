# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cuda.tile._backend import cpu
from cutile_cpu_utils import gelu
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import reduce_first_dim
from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _max_init_val(dtype, BLOCK_SIZE_N, **kwargs):
    return ct.full((BLOCK_SIZE_N,), float("-inf"), dtype=dtype)


def _max_first_dim(values, block, **kwargs):
    return ct.maximum(values, block)


def _max_last_dim(value, **kwargs):
    return value * 0.0


@ct.kernel
def _normalize_first_dim(Input, Output, N, BLOCK_SIZE_N: ct.Constant[int]):
    row_sum = ct.full((), 0, dtype=Input.dtype)
    for block in range(N // BLOCK_SIZE_N):
        values = ct.load(Input, index=(0, block), shape=(1, BLOCK_SIZE_N))
        row_sum += ct.sum(ct.reshape(values, (BLOCK_SIZE_N,)), axis=0)
    row_mean = row_sum / N
    for block in range(N // BLOCK_SIZE_N):
        values = ct.load(Input, index=(0, block), shape=(1, BLOCK_SIZE_N))
        values = gelu(ct.reshape(values, (BLOCK_SIZE_N,)) - row_mean)
        ct.store(
            Output,
            index=(0, block),
            tile=ct.reshape(values.astype(Output.dtype), (1, BLOCK_SIZE_N)),
        )


class Model(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.linear.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.linear.bias.data.to(dtype=x.dtype).contiguous()

        if self.max_dim == 0:
            res_mm = sfc_matmul(
                x,
                self._weight_packed,
                bias=self._bias,
                trunc_output=False,
                b_is_prepacked=True,
                c_is_owned=True,
                blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
            )
            res_max = reduce_first_dim(
                res_mm,
                out_dtype=res_mm.dtype,
                init_val=_max_init_val,
                reduction_op=_max_first_dim,
                keep_dim=True,
            )
            _, n = res_max.shape
            block_size_n = 256
            assert n % block_size_n == 0
            output = torch.empty((1, n), dtype=x.dtype)
            with cpu.compile_options({"assume_in_bounds": True}):
                ct.launch(
                    None,
                    (1,),
                    _normalize_first_dim,
                    (res_max, output, n, block_size_n),
                )
            return output

        assert self.max_dim == 1, "max_dim must be either 0 or 1"
        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            reduce_last_dim=True,
            reduction_post_op=_max_last_dim,
            keep_dim=True,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
