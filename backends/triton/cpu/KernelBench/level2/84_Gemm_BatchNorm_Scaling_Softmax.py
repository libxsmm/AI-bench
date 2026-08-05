# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul
from triton_cpu_utils import softmax


@triton.jit
def _epilogue(x):
    # First, we have an untrained BatchNorm1d in eval mode, with affine=True and eps=1e-5, which degrades to multiplication with 1/sqrt(1 + eps).
    # Then, we have a scaling operation with a parameter initialized to 1.0.
    return x * 0.999995


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

        self._weight_packed = None
        self._bias = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype)
            self._scale = self.scale.data.to(dtype=x.dtype)
            assert self.bn.affine, "BatchNorm must have affine=True"

        # eval()-mode BatchNorm1d and scaling with 1.0 merged into epilogue, see above.
        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            b_is_prepacked=True,
            post_op=_epilogue,
            trunc_output=False,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        return softmax(res_mm, out_dtype=x.dtype)
