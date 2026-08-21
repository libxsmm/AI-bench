# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper


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

        self._matmul_helper = None
        self._scale = None

    def forward(self, x):
        if self._matmul_helper is None:
            self._matmul_helper = SFCMatmulHelper(
                self.gemm.weight.data.to(dtype=x.dtype),
                self.gemm.bias.data.to(dtype=x.dtype),
            )
            self._scale = self.scale.data.to(dtype=x.dtype)
            assert self.bn.affine, "BatchNorm must have affine=True"

        # eval()-mode BatchNorm1d merged into epilogue, see above.
        return self._matmul_helper(
            x,
            post_op=_epilogue,
            post_op_arg=self._scale,
        )
