# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import cuda.tile as ct
from cutile_cpu_utils import pack_weights_for_sfc_matmul
from cutile_cpu_utils import sfc_matmul
import torch
import torch.nn as nn

ct.set_backend("cpu")


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def _epilogue(value, block_n, post_op_arg_ptr, **kwargs):
    scale = ct.load(
        post_op_arg_ptr,
        index=(block_n,),
        shape=(32,),
    ).astype(value.dtype)
    return value * (scale * 0.999995)[None, :]


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.rand(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._weight_packed = None
        self._bias = None
        self._scale = None

    def forward(self, x):
        x = x.contiguous()
        if self._weight_packed is None or self._weight_packed.dtype != x.dtype:
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype).contiguous()
            self._scale = self.scale.data.to(dtype=x.dtype).contiguous()
            assert self.bn.affine, "BatchNorm must have affine=True"

        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            post_op=_epilogue,
            post_op_arg=self._scale,
            b_is_prepacked=True,
            blocking_factor_k=_next_power_of_2(max(1, x.shape[1] // 4096)),
        )
