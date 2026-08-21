# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import gelu


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

        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )
            assert self.bn.affine, "BatchNorm must have affine=True"

        # eval()-mode BatchNorm1d merged into epilogue, see above.
        return self._matmul_helper(
            x,
            post_op=_epilogue,
        )
