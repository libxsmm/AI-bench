# ruff: noqa: E731
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative


import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton_cpu_utils import SFCMatmulHelper
from triton_cpu_utils import pack_weights_for_sfc_matmul
from triton_cpu_utils.sfc_matmul import _make_sfc_tensor


# Each program computes a (BLOCK_SIZE_M, GROUP_SIZE) output tile, i.e. the
# columns of exactly one GroupNorm group, so the normalization statistics can
# be computed in the GEMM epilogue. The weight is block-and-VNNI-packed with
# BLOCK_SIZE_N == GROUP_SIZE by `pack_weights_for_sfc_matmul`.
@triton.jit
def _fused_groupnorm_kernel(
    x_ptr,
    w_ptr,
    params_ptr,
    out_ptr,
    sfc_map_ptr,
    M,
    N,
    K,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    VNNI: tl.constexpr = 32 // w_ptr.type.element_ty.primitive_bitwidth

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_K = K // BLOCK_SIZE_K
    NUM_GROUPS = N // GROUP_SIZE

    pid = tl.program_id(axis=0)
    block_m = tl.load(sfc_map_ptr + 2 * pid)
    group = tl.load(sfc_map_ptr + 2 * pid + 1)

    # Blocked view of the row-major (M, K) activations, no packing copy.
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
        strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_K, K, 1),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K),
    )
    w_desc = tl.make_tensor_descriptor(
        base=w_ptr,
        shape=(NUM_GROUPS, BLOCKS_K, BLOCK_SIZE_K // VNNI, GROUP_SIZE * VNNI),
        strides=(GROUP_SIZE * K, BLOCK_SIZE_K * GROUP_SIZE, GROUP_SIZE * VNNI, 1),
        block_shape=(1, 1, BLOCK_SIZE_K // VNNI, GROUP_SIZE * VNNI),
    )
    params_desc = tl.make_tensor_descriptor(
        base=params_ptr,
        shape=(4, N),
        strides=(N, 1),
        block_shape=(1, GROUP_SIZE),
    )
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_SIZE_M, GROUP_SIZE),
    )

    acc = tl.zeros((BLOCK_SIZE_M, GROUP_SIZE), dtype=tl.float32)
    for block_k in range(BLOCKS_K):
        a = x_desc.load([block_m, block_k, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = w_desc.load([group, block_k, 0, 0]).reshape(
            (BLOCK_SIZE_K // VNNI, GROUP_SIZE * VNNI)
        )
        if VNNI > 1:
            b = tl.extra.cpu.vnni_decode(b)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    col = group * GROUP_SIZE
    linear_bias = params_desc.load([0, col]).reshape((GROUP_SIZE,))
    extra_bias = params_desc.load([1, col]).reshape((GROUP_SIZE,))
    norm_weight = params_desc.load([2, col]).reshape((GROUP_SIZE,))
    norm_bias = params_desc.load([3, col]).reshape((GROUP_SIZE,))

    y = acc + linear_bias[None, :]
    y = tl.sigmoid(y) * y + extra_bias[None, :]
    mean = tl.sum(y, axis=1) / GROUP_SIZE
    centered = y - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / GROUP_SIZE
    y = centered * tl.rsqrt(var + eps)[:, None]
    y = y * norm_weight[None, :] + norm_bias[None, :]
    out_desc.store([block_m * BLOCK_SIZE_M, col], y.to(out_ptr.type.element_ty))


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self._cache = None

    def _reference(self, x):
        return self.group_norm(torch.nn.functional.silu(self.matmul(x)) + self.bias)

    def forward(self, x):
        M, K = x.shape
        N = self.matmul.out_features
        group_size = N // self.num_groups
        if x.device.type != "cpu":
            return self._reference(x)
        try:
            BLOCK_SIZE_M, _, BLOCK_SIZE_K = SFCMatmulHelper.get_block_sizes(x.dtype)
        except ValueError:
            return self._reference(x)
        if (
            M % BLOCK_SIZE_M
            or K % BLOCK_SIZE_K
            or group_size < 16
            or group_size > 256
            or group_size & (group_size - 1)
        ):
            return self._reference(x)

        sources = (
            self.matmul.weight,
            self.matmul.bias,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
        )
        key = (
            tuple((param.data_ptr(), param._version) for param in sources),
            x.dtype,
        )
        if self._cache is None or self._cache[0] != key:
            weight = self.matmul.weight.detach().to(dtype=x.dtype).contiguous()
            packed = pack_weights_for_sfc_matmul(weight, group_size, BLOCK_SIZE_K)
            params = torch.stack(
                [param.detach().to(dtype=torch.float32) for param in sources[1:]]
            ).contiguous()
            self._cache = (key, packed, params)

        x = x.contiguous()
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        blocks_m = M // BLOCK_SIZE_M
        sfc_map = _make_sfc_tensor(blocks_m, self.num_groups)
        _fused_groupnorm_kernel[(blocks_m * self.num_groups,)](
            x,
            self._cache[1],
            self._cache[2],
            out,
            sfc_map,
            M,
            N,
            K,
            self.group_norm.eps,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE=group_size,
            assume_in_bounds=True,
        )
        return out
