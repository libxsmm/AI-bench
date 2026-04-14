# ruff: noqa: E731

import torch
import triton
import triton.language as tl
import torch.nn as nn

# -------------------------------------------------------------------
# Reference sizes / helpers
# -------------------------------------------------------------------

batch_size = 4096
input_size = 2048
hidden_size = 4096
output_size = 1024


def get_init_inputs():
    return [input_size, hidden_size, output_size]


def get_inputs():
    return [torch.rand(batch_size, input_size, dtype=torch.float16, device="xpu")]


# -------------------------------------------------------------------
# Triton Kernel 1: Fused Linear + Sigmoid
# Uses packed weights in [K, N] layout to avoid transpose in K-loop.
# Adds XPU-oriented configs and GROUP_SIZE_M swizzling for better locality.
# -------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=16,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=32,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_sigmoid_kernel_packed(
    x_ptr, w_t_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_width = GROUP_SIZE_M * num_pid_n
    group_id = pid // group_width
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % group_width) // group_size_m

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_t_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        x_vals = tl.load(x_bp, boundary_check=(0, 1))
        w_vals = tl.load(w_bp, boundary_check=(0, 1))
        acc += tl.dot(x_vals, w_vals)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    pos = acc >= 0
    out_pos = 1.0 / (1.0 + tl.exp(-acc))
    exp_acc = tl.exp(acc)
    out = tl.where(pos, out_pos, exp_acc / (1.0 + exp_acc))

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, out.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


# -------------------------------------------------------------------
# Retained reference kernel to preserve interface structure.
# -------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
    ],
    key=["In", "Out"],
)
@triton.jit
def _linear_logsumexp_fused_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, In, Out,
    stride_xm, stride_xk,
    stride_wn, stride_wi,
    stride_b,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= B:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m = tl.full((), -float("inf"), dtype=tl.float32)
    l = tl.zeros((), dtype=tl.float32)

    n_tiles = tl.cdiv(Out, BLOCK_N)
    k_tiles = tl.cdiv(In, BLOCK_K)

    inv_ln2 = 1.4426950408889634
    ln2 = 0.6931471805599453

    for nt in range(n_tiles):
        start_n = nt * BLOCK_N
        n_idx = start_n + offs_n
        n_mask = n_idx < Out

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for kt in range(k_tiles):
            start_k = kt * BLOCK_K
            k_idx = start_k + offs_k
            k_mask = k_idx < In

            x_vals = tl.load(
                x_ptr + row * stride_xm + k_idx * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)

            w_ptrs = w_ptr + n_idx[:, None] * stride_wn + k_idx[None, :] * stride_wi
            w_vals = tl.load(
                w_ptrs,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        b_vals = tl.load(b_ptr + n_idx * stride_b, mask=n_mask, other=0.0).to(tl.float32)
        acc += b_vals

        block_m = tl.max(acc, axis=0)
        m_new = tl.maximum(m, block_m)
        alpha = tl.math.exp2((m - m_new) * inv_ln2)
        sum_exp = tl.sum(tl.math.exp2((acc - m_new) * inv_ln2), axis=0)
        l = l * alpha + sum_exp
        m = m_new

    y_val = m + tl.math.log2(l) * ln2
    tl.store(y_ptr + row, y_val)


# -------------------------------------------------------------------
# Stage-2 optimized decomposition:
#   1) compute logits tile stats with larger row blocking and exp2 math
#   2) reduce stats across output tiles using stable merge
# Uses packed second-layer weights in [In, Out] = [K, N] layout
# and swizzled program ordering for better cache behavior.
# -------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=16,
        ),
    ],
    key=["B", "In", "Out"],
)
@triton.jit
def _linear_lse_tile_stats_block_kernel_packed(
    x_ptr, w_t_ptr, b_ptr,
    tile_max_ptr, tile_sum_ptr,
    B, In, Out,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_b,
    stride_tm_row, stride_tm_tile,
    stride_ts_row, stride_ts_tile,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B, BLOCK_M)
    num_pid_n = tl.cdiv(Out, BLOCK_N)

    group_width = GROUP_SIZE_M * num_pid_n
    group_id = pid // group_width
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % group_width) // group_size_m

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(B, In),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_t_ptr,
        shape=(In, Out),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, In, BLOCK_K):
        x_vals = tl.load(x_bp, boundary_check=(0, 1))
        w_vals = tl.load(w_bp, boundary_check=(0, 1))
        acc += tl.dot(x_vals, w_vals)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < B
    mask_n = offs_n < Out

    b_vals = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    inv_ln2 = 1.4426950408889634
    neg_inf = -float("inf")
    acc_masked = tl.where(mask_n[None, :], acc, neg_inf)
    m = tl.max(acc_masked, axis=1)
    s = tl.sum(
        tl.where(mask_n[None, :], tl.math.exp2((acc - m[:, None]) * inv_ln2), 0.0),
        axis=1,
    )

    tl.store(tile_max_ptr + offs_m * stride_tm_row + pid_n * stride_tm_tile, m, mask=mask_m)
    tl.store(tile_sum_ptr + offs_m * stride_ts_row + pid_n * stride_ts_tile, s, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_T": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_T": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_T": 16}, num_stages=2, num_warps=8),
    ],
    key=["B", "num_tiles"],
)
@triton.jit
def _reduce_lse_tiles_block_kernel(
    tile_max_ptr, tile_sum_ptr, y_ptr,
    B, num_tiles,
    stride_tm_row, stride_tm_tile,
    stride_ts_row, stride_ts_tile,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B

    inv_ln2 = 1.4426950408889634
    ln2 = 0.6931471805599453

    m = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for t0 in range(0, num_tiles, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < num_tiles

        tm = tl.load(
            tile_max_ptr + offs_m[:, None] * stride_tm_row + offs_t[None, :] * stride_tm_tile,
            mask=mask_m[:, None] & mask_t[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        ts = tl.load(
            tile_sum_ptr + offs_m[:, None] * stride_ts_row + offs_t[None, :] * stride_ts_tile,
            mask=mask_m[:, None] & mask_t[None, :],
            other=0.0,
        ).to(tl.float32)

        block_m = tl.max(tm, axis=1)
        block_l = tl.sum(ts * tl.math.exp2((tm - block_m[:, None]) * inv_ln2), axis=1)

        m_new = tl.maximum(m, block_m)
        l = l * tl.math.exp2((m - m_new) * inv_ln2) + block_l * tl.math.exp2((block_m - m_new) * inv_ln2)
        m = m_new

    tl.store(y_ptr + offs_m, m + tl.math.log2(l) * ln2, mask=mask_m)


# -------------------------------------------------------------------
# Top-level kernel wrapper
# -------------------------------------------------------------------

def kernel_function(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    if not all(isinstance(t, torch.Tensor) for t in (x, w1, b1, w2, b2)):
        raise TypeError("All inputs must be torch.Tensor")

    x_xpu = x if x.device.type == "xpu" and x.dtype == torch.float16 else x.to("xpu", dtype=torch.float16)
    w1_xpu = w1 if w1.device.type == "xpu" and w1.dtype == torch.float16 else w1.to("xpu", dtype=torch.float16)
    b1_xpu = b1 if b1.device.type == "xpu" and b1.dtype == torch.float16 else b1.to("xpu", dtype=torch.float16)
    w2_xpu = w2 if w2.device.type == "xpu" and w2.dtype == torch.float16 else w2.to("xpu", dtype=torch.float16)
    b2_xpu = b2 if b2.device.type == "xpu" and b2.dtype == torch.float16 else b2.to("xpu", dtype=torch.float16)

    x_xpu = x_xpu.contiguous()
    w1_xpu = w1_xpu.contiguous()
    b1_xpu = b1_xpu.contiguous()
    w2_xpu = w2_xpu.contiguous()
    b2_xpu = b2_xpu.contiguous()

    # Prepacked weights expected from Model.forward fast path when available.
    # Fallback keeps kernel_function correct if called directly.
    w1_t_xpu = w1_xpu.transpose(0, 1).contiguous()
    w2_t_xpu = w2_xpu.transpose(0, 1).contiguous()

    B, In = x_xpu.shape
    H, In_w1 = w1_xpu.shape
    if In != In_w1:
        raise ValueError("x.shape[1] must match w1.shape[1]")
    if b1_xpu.numel() != H:
        raise ValueError("b1 length must match hidden dim")

    O, H_w2 = w2_xpu.shape
    if H != H_w2:
        raise ValueError("w2.shape[1] must match hidden dim")
    if b2_xpu.numel() != O:
        raise ValueError("b2 length must match output dim")

    hidden = torch.empty((B, H), dtype=torch.float16, device="xpu")
    y = torch.empty((B,), dtype=torch.float32, device="xpu")

    grid1 = (triton.cdiv(B, 128) * triton.cdiv(H, 128),)
    _linear_sigmoid_kernel_packed[grid1](
        x_xpu, w1_t_xpu, b1_xpu, hidden,
        B, H, In,
        x_xpu.stride(0), x_xpu.stride(1),
        w1_t_xpu.stride(0), w1_t_xpu.stride(1),
        hidden.stride(0), hidden.stride(1),
    )

    block_n_stats = 128
    num_tiles = triton.cdiv(O, block_n_stats)
    tile_max = torch.empty((B, num_tiles), dtype=torch.float32, device="xpu")
    tile_sum = torch.empty((B, num_tiles), dtype=torch.float32, device="xpu")

    grid2 = (triton.cdiv(B, 16) * num_tiles,)
    _linear_lse_tile_stats_block_kernel_packed[grid2](
        hidden, w2_t_xpu, b2_xpu,
        tile_max, tile_sum,
        B, H, O,
        hidden.stride(0), hidden.stride(1),
        w2_t_xpu.stride(0), w2_t_xpu.stride(1),
        b2_xpu.stride(0),
        tile_max.stride(0), tile_max.stride(1),
        tile_sum.stride(0), tile_sum.stride(1),
    )

    grid3 = (triton.cdiv(B, 16),)
    _reduce_lse_tiles_block_kernel[grid3](
        tile_max, tile_sum, y,
        B, num_tiles,
        tile_max.stride(0), tile_max.stride(1),
        tile_sum.stride(0), tile_sum.stride(1),
    )

    return y.to(x.dtype)


# -------------------------------------------------------------------
# KernelBench Model wrapper
# Cache packed weights once to avoid per-forward transpose+contiguous cost.
# -------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, device="cpu", dtype=torch.float16)
        self.linear2 = nn.Linear(hidden_size, output_size, device="cpu", dtype=torch.float16)
        self._moved_to_xpu = False
        self._w1_t_packed = None
        self._w2_t_packed = None

    def _ensure_xpu_and_packed(self):
        if not self._moved_to_xpu:
            self.linear1.weight.data = self.linear1.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.linear1.bias.data = self.linear1.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.linear2.weight.data = self.linear2.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.linear2.bias.data = self.linear2.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self._w1_t_packed = self.linear1.weight.transpose(0, 1).contiguous()
            self._w2_t_packed = self.linear2.weight.transpose(0, 1).contiguous()
            self._moved_to_xpu = True

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)

        self._ensure_xpu_and_packed()

        # Inline wrapper to actually use cached packed weights and avoid repeated packing.
        x_xpu = x.contiguous()
        b1_xpu = self.linear1.bias
        b2_xpu = self.linear2.bias
        w1_t_xpu = self._w1_t_packed
        w2_t_xpu = self._w2_t_packed

        B, In = x_xpu.shape
        H = self.linear1.weight.shape[0]
        O = self.linear2.weight.shape[0]

        hidden = torch.empty((B, H), dtype=torch.float16, device="xpu")
        y = torch.empty((B,), dtype=torch.float32, device="xpu")

        grid1 = (triton.cdiv(B, 128) * triton.cdiv(H, 128),)
        _linear_sigmoid_kernel_packed[grid1](
            x_xpu, w1_t_xpu, b1_xpu, hidden,
            B, H, In,
            x_xpu.stride(0), x_xpu.stride(1),
            w1_t_xpu.stride(0), w1_t_xpu.stride(1),
            hidden.stride(0), hidden.stride(1),
        )

        block_n_stats = 128
        num_tiles = triton.cdiv(O, block_n_stats)
        tile_max = torch.empty((B, num_tiles), dtype=torch.float32, device="xpu")
        tile_sum = torch.empty((B, num_tiles), dtype=torch.float32, device="xpu")

        grid2 = (triton.cdiv(B, 16) * num_tiles,)
        _linear_lse_tile_stats_block_kernel_packed[grid2](
            hidden, w2_t_xpu, b2_xpu,
            tile_max, tile_sum,
            B, H, O,
            hidden.stride(0), hidden.stride(1),
            w2_t_xpu.stride(0), w2_t_xpu.stride(1),
            b2_xpu.stride(0),
            tile_max.stride(0), tile_max.stride(1),
            tile_sum.stride(0), tile_sum.stride(1),
        )

        grid3 = (triton.cdiv(B, 16),)
        _reduce_lse_tiles_block_kernel[grid3](
            tile_max, tile_sum, y,
            B, num_tiles,
            tile_max.stride(0), tile_max.stride(1),
            tile_sum.stride(0), tile_sum.stride(1),
        )

        return y.to(x.dtype)