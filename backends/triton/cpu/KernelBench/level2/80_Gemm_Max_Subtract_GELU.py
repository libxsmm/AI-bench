# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import gelu
from triton_cpu_utils import reduce_first_dim


@triton.jit
def _max_init_val(dtype, BLOCK_SIZE_N):
    return tl.full([BLOCK_SIZE_N], float("-inf"), dtype=dtype)


@triton.jit
def _max_first_dim(vals, x, BLOCK_SIZE_N):
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
def _max_epi_last_dim(val, M, N):
    # mean of a single element -> the max_dim=1 benchmark config is a degenerate case that just returns zeros.
    return val * 0.0


class Model(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.linear.weight.data.to(dtype=x.dtype),
                self.linear.bias.data.to(dtype=x.dtype),
            )

        if self.max_dim == 0:
            res_mm = self._matmul_helper(
                x,
                trunc_output=False,
                c_is_owned=True,
            )

            res_max = reduce_first_dim(
                res_mm,
                out_dtype=res_mm.dtype,  # keep it in f32
                init_val=_max_init_val,
                reduction_op=_max_first_dim,
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
        return self._matmul_helper(
            x,
            reduce_last_dim=True,
            reduction_post_op=_max_epi_last_dim,
            keep_dim=True,
        )
