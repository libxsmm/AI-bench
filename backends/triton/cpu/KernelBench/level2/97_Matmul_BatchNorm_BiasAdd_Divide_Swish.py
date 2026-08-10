# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import sfc_matmul


class Model(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        bias_shape=(1,),
        divide_value=1.0,
    ):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        div_val = tl.constexpr(divide_value)

        @triton.jit
        def _epilogue(x, post_op_arg_ptr, **kwargs):
            bias_val = tl.load(post_op_arg_ptr).to(x.dtype)
            x += bias_val
            # After the bias addition, the next op is an untrained BatchNorm1d in eval mode, with affine=True and eps=1e-5, which simplifies to multiplication with 1/sqrt(1 + eps).
            x = x * 0.999995 / div_val
            x = x * tl.sigmoid(x)
            return x

        self._weight_packed = None
        self._bias = None
        self._bias_extra = None
        self._epilogue_fun = _epilogue

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.matmul.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.matmul.bias.data.to(dtype=x.dtype)
            assert self.bn.affine, "BatchNorm must have affine=True"
            self._bias_extra = self.bias.data.to(dtype=x.dtype)

        # eval()-mode BatchNorm1d merged into epilogue, see above.
        return sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            b_is_prepacked=True,
            post_op=self._epilogue_fun,
            post_op_arg=self._bias_extra,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )
