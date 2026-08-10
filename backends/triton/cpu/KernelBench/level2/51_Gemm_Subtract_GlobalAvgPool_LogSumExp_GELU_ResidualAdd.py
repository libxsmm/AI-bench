# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul


@triton.jit
def _epilogue(x, block_n, post_op_arg_ptr, N, BLOCK_SIZE_N, **kwargs):
    desc = tl.make_tensor_descriptor(
        base=post_op_arg_ptr,
        shape=(N,),
        strides=(1,),
        block_shape=(BLOCK_SIZE_N,),
    )
    neg_bias = desc.load([block_n * BLOCK_SIZE_N]).to(x.dtype)
    return x - neg_bias[None, :]


@triton.jit
def _red(val, x, **kwargs):
    return val + tl.sum(x, axis=0)


@triton.jit
def _red_epi(val, N, **kwargs):
    return gelu(val / N)


@triton.jit
def _residual_add(col_vec_ptr, orig_mat_ptr, out_mat_ptr, M, K, BS_K: tl.constexpr):
    m = tl.program_id(0)
    orig_mat_desc = tl.make_tensor_descriptor(
        base=orig_mat_ptr,
        shape=(M, K),
        strides=(K, 1),
        block_shape=(1, BS_K),
    )
    out_mat_desc = tl.make_tensor_descriptor(
        base=out_mat_ptr,
        shape=(M, K),
        strides=(K, 1),
        block_shape=(1, BS_K),
    )

    row_mean = tl.load(col_vec_ptr + m)
    for k in range(0, K, BS_K):
        orig_row = orig_mat_desc.load([m, k]).reshape([BS_K])
        out_row = orig_row + row_mean
        out_mat_desc.store(
            [m, k], out_row.to(out_mat_ptr.type.element_ty).reshape([1, BS_K])
        )


batch_size = 2048
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

        self._weight_packed = None
        self._bias = None
        self._neg_bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype)
            self._neg_bias = self.subtract.data.to(dtype=x.dtype)

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            b_is_prepacked=True,
            post_op=_epilogue,
            post_op_arg=self._neg_bias,
            trunc_output=False,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        res_mean = reduce_last_dim(
            res_mm,
            out_dtype=res_mm.dtype,
            reduction=_red,
            post_op=_red_epi,
            keep_dim=True,
        )

        # The logsumexp after the first reduction is a no-op, so we fuse the gelu right away.

        res = torch.empty_like(x)
        BLOCK_SIZE_K = 256
        assert x.shape[1] % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K"
        _residual_add[(x.shape[0],)](
            col_vec_ptr=res_mean,
            orig_mat_ptr=x,
            out_mat_ptr=res,
            M=x.shape[0],
            K=x.shape[1],
            BS_K=BLOCK_SIZE_K,
        )

        return res
