# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_first_dim
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul


@triton.jit
def _max_first_dim(vals, x, **kwargs):
    return tl.maximum(vals, x)


@triton.jit
def _kernel_first_dim(inp_ptr, out_ptr, N, BLOCK_SIZE_N: tl.constexpr):
    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[1, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[1, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    row_sum = tl.zeros([], dtype=inp_ptr.type.element_ty)
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([0, n]).reshape([BLOCK_SIZE_N])
        row_sum += tl.sum(x, axis=0)

    row_mean = row_sum / N
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([0, n]).reshape([BLOCK_SIZE_N])
        x = x - row_mean
        x = gelu(x)
        out_desc.store([0, n], x.to(out_ptr.type.element_ty).reshape([1, BLOCK_SIZE_N]))


@triton.jit
def _max_last_dim(val, x, **kwargs):
    return tl.maximum(val, tl.max(x))


@triton.jit
def _max_epi_last_dim(val, **kwargs):
    # mean of a single element -> the max_dim=1 benchmark config is a degenerate case that just returns zeros.
    return gelu(val - val)


class Model(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

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

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            trunc_output=False,
            b_is_prepacked=True,
            c_is_owned=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        if self.max_dim == 0:
            res_max = reduce_first_dim(
                res_mm,
                out_dtype=res_mm.dtype,  # keep it in f32
                reduction=_max_first_dim,
                keep_dim=True,
            )
            M, N = res_max.shape
            BLOCK_SIZE_N = 256  # AVX-512 optimized block size
            assert M == 1 and N % BLOCK_SIZE_N == 0, (
                "N must be divisible by BLOCK_SIZE_N"
            )
            out = torch.empty((1, N), dtype=x.dtype)
            _kernel_first_dim[(1,)](
                res_max,
                out,
                N,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                assume_in_bounds=True,
            )
            return out

        assert self.max_dim == 1, "max_dim must be either 0 or 1"
        res_max = reduce_last_dim(
            res_mm,
            out_dtype=x.dtype,
            reduction=_max_last_dim,
            post_op=_max_epi_last_dim,
            keep_dim=True,
        )

        return res_max
