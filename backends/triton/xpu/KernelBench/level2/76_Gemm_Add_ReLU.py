# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_bias_relu_kernel(
    a_ptr,
    b_t_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_SIZE_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_SIZE_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_t_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.max_contiguous(offs_n, BLOCK_N)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    acc += bias[None, :]
    acc = tl.maximum(acc, 0.0)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 32,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 64,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 128,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 256,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 32,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 64,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 128,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 32,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 64,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 1,
                "NUM_PROGS": 128,
            },
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 32,
            },
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 64,
            },
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 128,
            },
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 32,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 64,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 128,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 32,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 64,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "NUM_PROGS": 128,
            },
            num_warps=16,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_bias_relu_persistent_kernel(
    a_ptr,
    b_t_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    tile_id = pid
    while tile_id < num_tiles:
        group_size = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // group_size
        first_pid_m = group_id * GROUP_SIZE_M
        group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_in_group = tile_id % group_size
        pid_m = first_pid_m + (pid_in_group % group_m)
        pid_n = pid_in_group // group_m

        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_t_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc=acc)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.max_contiguous(offs_n, BLOCK_N)
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

        acc += bias[None, :]
        acc = tl.maximum(acc, 0.0)

        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

        tile_id += NUM_PROGS


def _get_num_xpu_workers():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return 32
    try:
        if hasattr(torch.xpu, "get_device_capability"):
            cap = torch.xpu.get_device_capability()
            if isinstance(cap, dict):
                for key in (
                    "gpu_subslice_count",
                    "subslice_count",
                    "max_compute_units",
                    "gpu_eu_count",
                ):
                    val = cap.get(key, None)
                    if isinstance(val, int) and val > 0:
                        return val
        if hasattr(torch.xpu, "get_device_properties"):
            props = torch.xpu.get_device_properties(torch.xpu.current_device())
            for key in ("subslice_count", "max_compute_units", "multi_processor_count"):
                if hasattr(props, key):
                    val = getattr(props, key)
                    if isinstance(val, int) and val > 0:
                        return val
    except Exception:
        pass
    return 32


def _select_num_progs_cap(total_tiles: int):
    hw = _get_num_xpu_workers()
    cap = max(1, min(total_tiles, hw))
    if cap >= 256:
        return 256
    if cap >= 128:
        return 128
    if cap >= 64:
        return 64
    if cap >= 32:
        return 32
    return cap


def kernel_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    packed_weight_t: torch.Tensor = None,
):
    assert (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and isinstance(bias, torch.Tensor)
    )
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1
    assert x.shape[1] == weight.shape[1], "Incompatible shapes"
    assert bias.numel() == weight.shape[0], "Bias length mismatch"

    x_xpu = (
        x
        if (x.device.type == "xpu" and x.dtype == torch.float16 and x.is_contiguous())
        else x.to(device="xpu", dtype=torch.float16).contiguous()
    )
    weight_xpu = (
        weight
        if (
            weight.device.type == "xpu"
            and weight.dtype == torch.float16
            and weight.is_contiguous()
        )
        else weight.to(device="xpu", dtype=torch.float16).contiguous()
    )
    bias_xpu = (
        bias
        if (
            bias.device.type == "xpu"
            and bias.dtype == torch.float16
            and bias.is_contiguous()
        )
        else bias.to(device="xpu", dtype=torch.float16).contiguous()
    )

    if packed_weight_t is not None:
        weight_t_xpu = (
            packed_weight_t
            if (
                packed_weight_t.device.type == "xpu"
                and packed_weight_t.dtype == torch.float16
                and packed_weight_t.is_contiguous()
            )
            else packed_weight_t.to(device="xpu", dtype=torch.float16).contiguous()
        )
    else:
        weight_t_xpu = weight_xpu.transpose(0, 1).contiguous()

    M, K = x_xpu.shape
    N = weight_xpu.shape[0]

    out = torch.empty((M, N), device="xpu", dtype=torch.float16)

    stride_am, stride_ak = x_xpu.stride()
    stride_bk, stride_bn = weight_t_xpu.stride()
    stride_cm, stride_cn = out.stride()

    total_tiles = triton.cdiv(M, 256) * triton.cdiv(N, 256)
    num_progs_cap = _select_num_progs_cap(total_tiles)

    # Use persistent scheduling when there are enough tiles to amortize looping.
    # Fall back to original kernel for very small grids to avoid persistent overhead.
    if total_tiles >= 8 and num_progs_cap >= 1:
        grid = lambda meta: (min(meta["NUM_PROGS"], num_progs_cap),)
        _gemm_bias_relu_persistent_kernel[grid](
            x_xpu,
            weight_t_xpu,
            bias_xpu,
            out,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        _gemm_bias_relu_kernel[grid](
            x_xpu,
            weight_t_xpu,
            bias_xpu,
            out,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
        )
    return out


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias_shape = bias_shape
        self._xpu_ready = False
        self._packed_weight_t = None
        self._packed_weight_version = None

    def _ensure_xpu_params(self):
        moved = False
        if (
            self.gemm.weight.device.type != "xpu"
            or self.gemm.weight.dtype != torch.float16
            or not self.gemm.weight.is_contiguous()
        ):
            self.gemm.weight.data = self.gemm.weight.data.to(
                device="xpu", dtype=torch.float16
            ).contiguous()
            moved = True
        if self.gemm.bias is not None and (
            self.gemm.bias.device.type != "xpu"
            or self.gemm.bias.dtype != torch.float16
            or not self.gemm.bias.is_contiguous()
        ):
            self.gemm.bias.data = self.gemm.bias.data.to(
                device="xpu", dtype=torch.float16
            ).contiguous()
            moved = True

        current_version = self.gemm.weight._version
        if (
            (not self._xpu_ready)
            or moved
            or (self._packed_weight_t is None)
            or (self._packed_weight_version != current_version)
        ):
            self._packed_weight_t = self.gemm.weight.transpose(0, 1).contiguous()
            self._packed_weight_version = current_version
            self._xpu_ready = True

    def forward(self, x):
        self._ensure_xpu_params()
        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to(device="xpu", dtype=torch.float16).contiguous()
        return kernel_function(
            x, self.gemm.weight, self.gemm.bias, self._packed_weight_t
        )
