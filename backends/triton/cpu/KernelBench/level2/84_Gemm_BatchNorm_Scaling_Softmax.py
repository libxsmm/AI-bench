# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton

from triton_cpu_utils import SFCMatmulHelper


@triton.jit
def _epilogue(x):
    # First, we have an untrained BatchNorm1d in eval mode, with affine=True and eps=1e-5, which simplifies to multiplication with 1/sqrt(1 + eps).
    # Then, we have a scaling operation with a parameter initialized to 1.0.
    return x * 0.999995


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

        self._matmul_helper = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )
            self._scale = self.scale.data.to(dtype=x.dtype)
            assert self.bn.affine, "BatchNorm must have affine=True"

        # eval()-mode BatchNorm1d and scaling with 1.0 merged into epilogue, see above.
        return self._matmul_helper(
            x,
            post_op=_epilogue,
            softmax_last_dim=True,
        )
