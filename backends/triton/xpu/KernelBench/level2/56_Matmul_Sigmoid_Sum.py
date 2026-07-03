# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_splitk_gemm_configs():
    return [
        # Small / fallback tiles
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        # Suggested / balanced XPU configs
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        # Broader swizzle sweep
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        # Larger N tiles
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=2,
        ),
        # Mandatory large-tile XPU coverage
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
    ]


def _get_reduce_configs():
    return [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "CHUNK_N": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "CHUNK_N": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "CHUNK_N": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "CHUNK_N": 128}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "CHUNK_N": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "CHUNK_N": 128}, num_warps=32, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512, "CHUNK_N": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512, "CHUNK_N": 128}, num_warps=32, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "CHUNK_N": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "CHUNK_N": 128}, num_warps=32, num_stages=2
        ),
    ]


@triton.autotune(
    configs=_get_splitk_gemm_configs(),
    key=["B", "I", "H"],
)
@triton.jit
def _fused_linear_sigmoid_sum_kernel_splitk(
    x_ptr,
    w_ptr,
    b_ptr,
    partial_ptr,
    B,
    I,
    H,
    stride_xb,
    stride_xi,
    stride_wh,
    stride_wi,
    stride_pb,
    stride_ph,
    stride_ps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)

    num_pid_m = tl.cdiv(B, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)

    if GROUP_SIZE_M > 1 and num_pid_m > 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < B
    mask_n = offs_n < H
    tl.max_contiguous(offs_n, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_per_split = tl.cdiv(I, SPLIT_K)
    k_start = pid_sk * k_per_split
    k_end = tl.minimum(k_start + k_per_split, I)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(B, I),
        strides=(stride_xb, stride_xi),
        offsets=(pid_m * BLOCK_M, k_start),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(I, H),
        strides=(stride_wi, stride_wh),
        offsets=(k_start, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    for _ in range(0, tl.cdiv(k_end - k_start, BLOCK_K)):
        x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        w_tile = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(x_tile, w_tile, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    partial_ptrs = (
        partial_ptr
        + offs_m[:, None, None].to(tl.int64) * stride_pb
        + offs_n[None, :, None].to(tl.int64) * stride_ph
        + pid_sk.to(tl.int64) * stride_ps
    )
    tl.store(
        partial_ptrs,
        acc[:, :, None].to(partial_ptr.dtype.element_ty),
        mask=mask_m[:, None, None] & mask_n[None, :, None],
    )


@triton.autotune(
    configs=_get_reduce_configs(),
    key=["B", "H"],
)
@triton.jit
def _reduce_sigmoid_sum_kernel_streamk(
    partial_ptr,
    b_ptr,
    y_ptr,
    B,
    H,
    stride_pb,
    stride_ph,
    stride_ps,
    stride_yb,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_N: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    LOG2E = 1.4426950408889634

    for n_start in range(0, H, BLOCK_N):
        for n0 in tl.static_range(0, BLOCK_N, CHUNK_N):
            offs_n = n_start + n0 + tl.arange(0, CHUNK_N)
            mask_n = offs_n < H
            tl.max_contiguous(offs_n, CHUNK_N)

            acc = tl.zeros((BLOCK_M, CHUNK_N), dtype=tl.float32)
            for sk in tl.static_range(0, SPLIT_K):
                ptrs = (
                    partial_ptr
                    + offs_m[:, None].to(tl.int64) * stride_pb
                    + offs_n[None, :].to(tl.int64) * stride_ph
                    + tl.full((), sk, tl.int64) * stride_ps
                )
                acc += tl.load(
                    ptrs,
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0,
                ).to(tl.float32)

            b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc = acc + b_vals[None, :]
            s = 1.0 / (1.0 + tl.math.exp2(-acc * LOG2E))
            row_sum += tl.sum(s, axis=1)

    y_ptrs = y_ptr + offs_m.to(tl.int64) * stride_yb
    tl.store(y_ptrs, row_sum.to(y_ptr.dtype.element_ty), mask=mask_m)


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert (
        isinstance(x, torch.Tensor)
        and isinstance(w, torch.Tensor)
        and isinstance(b, torch.Tensor)
    ), "x, w, b must be torch.Tensors"
    assert x.dtype == w.dtype == b.dtype == torch.float16, "Only float16 is supported"

    x_xpu = x if x.device.type == "xpu" else x.to("xpu", dtype=torch.float16)
    w_xpu = w if w.device.type == "xpu" else w.to("xpu", dtype=torch.float16)
    b_xpu = b if b.device.type == "xpu" else b.to("xpu", dtype=torch.float16)

    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()
    if not w_xpu.is_contiguous():
        w_xpu = w_xpu.contiguous()
    if not b_xpu.is_contiguous():
        b_xpu = b_xpu.contiguous()

    B, I = x_xpu.shape
    H, Iw = w_xpu.shape
    Hb = b_xpu.shape[0]
    assert I == Iw, f"Incompatible x and w dims: {I} vs {Iw}"
    assert H == Hb, f"Incompatible w and b dims: {H} vs {Hb}"

    y = torch.empty((B, 1), device=x_xpu.device, dtype=x_xpu.dtype)

    SPLIT_K = 8
    partial = torch.empty((B, H, SPLIT_K), device=x_xpu.device, dtype=torch.float16)

    grid_main = lambda META: (
        triton.cdiv(B, META["BLOCK_M"]) * triton.cdiv(H, META["BLOCK_N"]),
        SPLIT_K,
    )
    _fused_linear_sigmoid_sum_kernel_splitk[grid_main](
        x_xpu,
        w_xpu,
        b_xpu,
        partial,
        B,
        I,
        H,
        x_xpu.stride(0),
        x_xpu.stride(1),
        w_xpu.stride(0),
        w_xpu.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        SPLIT_K=SPLIT_K,
        grf_mode="auto",
    )

    grid_reduce = lambda META: (triton.cdiv(B, META["BLOCK_M"]),)
    _reduce_sigmoid_sum_kernel_streamk[grid_reduce](
        partial,
        b_xpu,
        y,
        B,
        H,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        y.stride(0),
        SPLIT_K=SPLIT_K,
        grf_mode="auto",
    )

    return y


batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._weights_on_xpu = False

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        elif not x.is_contiguous():
            x = x.contiguous()

        if not self._weights_on_xpu:
            self.linear.weight.data = self.linear.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
            self.linear.bias.data = self.linear.bias.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
            self._weights_on_xpu = True
        else:
            if (
                self.linear.weight.device.type != "xpu"
                or self.linear.weight.dtype != torch.float16
            ):
                self.linear.weight.data = self.linear.weight.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()
            elif not self.linear.weight.is_contiguous():
                self.linear.weight.data = self.linear.weight.data.contiguous()

            if (
                self.linear.bias.device.type != "xpu"
                or self.linear.bias.dtype != torch.float16
            ):
                self.linear.bias.data = self.linear.bias.data.to(
                    "xpu", dtype=torch.float16
                ).contiguous()
            elif not self.linear.bias.is_contiguous():
                self.linear.bias.data = self.linear.bias.data.contiguous()

        return kernel_function(x, self.linear.weight, self.linear.bias)
