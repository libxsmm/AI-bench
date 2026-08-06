# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import groupnorm
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils import reduce_last_dim
from triton_cpu_utils import sfc_matmul


@triton.jit
def _min_red(val, x, BLOCK_SIZE_N: tl.constexpr):
    return tl.minimum(val, tl.min(x))


@triton.jit
def _bcast_bias_kernel(bias_ptr, red_ptr, out_ptr, B, R, BLOCK_SIZE_R: tl.constexpr):
    red_desc = tl.make_tensor_descriptor(
        red_ptr,
        shape=[R],
        strides=[1],
        block_shape=[BLOCK_SIZE_R],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[B, R],
        strides=[R, 1],
        block_shape=[1, BLOCK_SIZE_R],
    )
    b = tl.program_id(0)
    bias = tl.load(bias_ptr + b)
    for r in range(0, R, BLOCK_SIZE_R):
        red_val = red_desc.load([r])
        out_val = red_val + bias
        out_desc.store(
            [b, r], out_val.to(out_ptr.type.element_ty).reshape([1, BLOCK_SIZE_R])
        )


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self.num_groups = num_groups
        self._weight_packed = None
        self._bias = None
        self._bias_extra = None

    def forward(self, x):
        if self._weight_packed is None:
            # AMX-optimized block size
            self._weight_packed = pack_weights_for_sfc_matmul(
                self.gemm.weight.data.to(dtype=x.dtype),
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
            )
            self._bias = self.gemm.bias.data.to(dtype=x.dtype)
            assert self.group_norm.affine, "GroupNorm must have affine=True"
            self._bias_extra = self.bias.data.to(dtype=x.dtype)
            assert self._bias_extra.shape[-2:] == (1, 1), (
                "Expecting reduction result to be broadcasted once per bias element"
            )

        res_mm = sfc_matmul(
            x,
            self._weight_packed,
            bias=self._bias,
            b_is_prepacked=True,
            trunc_output=False,
            blocking_factor_k=triton.next_power_of_2(max(1, x.shape[1] // 4096)),
        )

        res_gn = groupnorm(
            res_mm,
            out_dtype=res_mm.dtype,
            num_groups=self.num_groups,
            eps=self.group_norm.eps,
            try_inplace=True,
        )

        res_red = reduce_last_dim(
            res_gn,
            out_dtype=res_gn.dtype,
            reduction=_min_red,
            keep_dim=True,
        )

        res_shape = torch.broadcast_shapes(res_red.shape, self._bias_extra.shape)
        res = torch.empty(res_shape, dtype=x.dtype, device=x.device)
        n_bias_elements = torch.prod(
            torch.tensor(self._bias_extra.shape[:-2], dtype=torch.int32)
        ).item()
        BLOCK_SIZE_R = 256  # AVX-512 optimized block size
        assert res_red.shape[0] % BLOCK_SIZE_R == 0, (
            "Number of rows in reduction result must be divisible by BLOCK_SIZE_R"
        )
        _bcast_bias_kernel[(n_bias_elements,)](
            self._bias_extra,
            res_red,
            res,
            n_bias_elements,
            res_red.shape[0],
            BLOCK_SIZE_R=BLOCK_SIZE_R,
            assume_in_bounds=True,
        )
        return res
