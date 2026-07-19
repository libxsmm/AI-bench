# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


@triton.jit
def _reduce_kernel(inp_ptr, out_ptr, scaling_factor, M, N, BLOCK_SIZE_N: tl.constexpr):
    inp_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    row_sum = tl.zeros([], dtype=inp_ptr.type.element_ty)
    m = tl.program_id(0)
    for n in range(0, N, BLOCK_SIZE_N):
        x = inp_desc.load([m, n]).reshape([BLOCK_SIZE_N])
        row_sum += tl.sum(x, axis=0)
    row_sum *= scaling_factor

    tl.store(out_ptr + m, tl.cast(row_sum, out_ptr.type.element_ty))


batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

        @triton.jit
        def _gemm_epilogue(x):
            x /= 2.0
            return x

        self._gemm_epilogue_fun = _gemm_epilogue
        self._weight_packed = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            trunc_output=False,
            post_op=self._gemm_epilogue_fun,
            b_is_prepacked=True,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        M, N = res_mm.shape
        res_red = torch.empty((M, 1), dtype=x.dtype)
        BLOCK_SIZE_N = 256  # AVX-512 optimized block size
        assert N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N"
        _reduce_kernel[(M,)](
            res_mm,
            res_red,
            self.scaling_factor,
            M,
            N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            assume_in_bounds=True,
        )

        return res_red
