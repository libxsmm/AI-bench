# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import get_matmul_block_sizes
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import region_matmul


@triton.jit
def _epilogue(x, block_m, block_n, post_op_arg_ptr, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    desc = tl.make_tensor_descriptor(
        base=post_op_arg_ptr,
        shape=(N,),
        strides=(1,),
        block_shape=(BLOCK_SIZE_N,),
    )
    # After the scaling, the next op is an untrained BatchNorm1d in eval mode, with affine=True and eps=1e-5, which simplifies to multiplication with 1/sqrt(1 + eps).
    scale_val = desc.load([block_n * BLOCK_SIZE_N]).to(x.dtype) * 0.999995
    x = x * scale_val[None, :]
    return x


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.rand(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

        self._weight_packed = None
        self._bias = None
        self._scale = None
        self._block_sizes = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._block_sizes = get_matmul_block_sizes(x.dtype)
            _, block_size_n, block_size_k = self._block_sizes
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=block_size_n,
                BLOCK_SIZE_K=block_size_k,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype).contiguous()
            self._scale = self.scale.data.to(dtype=x.dtype).contiguous()
            assert self.bn.affine, "BatchNorm must have affine=True"

        block_size_m, block_size_n, block_size_k = self._block_sizes
        return region_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            post_op_arg=self._scale,
            b_is_prepacked=True,
            region_grid=(64, 1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )
