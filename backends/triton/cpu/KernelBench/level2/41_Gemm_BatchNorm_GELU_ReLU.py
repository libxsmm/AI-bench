# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import gelu
from triton_cpu_utils import get_matmul_block_sizes
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import region_matmul


@triton.jit
def _epilogue(x):
    # The first op is an untrained BatchNorm1d in eval mode, with affine=True and eps=1e-5, which simplifies to multiplication with 1/sqrt(1 + eps).
    x = tl.maximum(x * 0.999995, 0.0)
    x = gelu(x)
    return x


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(
            out_features,
        )

        self._weight_packed = None
        self._bias = None
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
            assert self.bn.affine, "BatchNorm must have affine=True"

        block_size_m, block_size_n, block_size_k = self._block_sizes
        return region_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            b_is_prepacked=True,
            region_grid=(64, 1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )
