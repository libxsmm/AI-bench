# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# =============================================================================
# Problem sizes / helpers
# =============================================================================
batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]


# =============================================================================
# Original Triton kernel preserved for benchmark compatibility/reference.
# =============================================================================
@triton.jit
def _linear_dropout_softmax_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    K,
    C,
    stride_xn,
    stride_xk,
    stride_wm,
    stride_wk,
    stride_on,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    arange_n = tl.arange(0, BLOCK_N)
    arange_k = tl.arange(0, BLOCK_K)

    for start_n in tl.range(0, C, BLOCK_N):
        offs_n = start_n + arange_n
        mask_n = offs_n < C
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            offs_k = start_k + arange_k
            mask_k = offs_k < K
            a_ptrs = x_ptr + (offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xk)
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_ptrs = w_ptr + (offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)

        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

        tile_max = tl.max(acc, axis=1)
        new_m = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - new_m)
        exp_tile = tl.exp(acc - new_m[:, None])
        l_i = l_i * alpha + tl.sum(exp_tile, axis=1)
        m_i = new_m

    for start_n in tl.range(0, C, BLOCK_N):
        offs_n = start_n + arange_n
        mask_n = offs_n < C
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            offs_k = start_k + arange_k
            mask_k = offs_k < K
            a_ptrs = x_ptr + (offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xk)
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_ptrs = w_ptr + (offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)

        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

        probs = tl.exp(acc - m_i[:, None]) / l_i[:, None]
        out_ptrs = out_ptr + (offs_m[:, None] * stride_on + offs_n[None, :] * stride_oc)
        tl.store(
            out_ptrs,
            probs.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _softmax_autotune_configs():
    configs = []

    # Row-reduction kernel search space: vary scan width and execution params.
    for block_n in (256, 512, 1024, 2048):
        for num_warps in (4, 8, 16):
            for num_stages in (2, 3, 4):
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_N": block_n,
                            "LOG2E": 1.4426950408889634,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    # XPU-focused high-warp configs for long rows.
    for block_n in (256, 512, 1024, 2048):
        for num_stages in (2, 3, 4):
            configs.append(
                triton.Config(
                    {
                        "BLOCK_N": block_n,
                        "LOG2E": 1.4426950408889634,
                    },
                    num_warps=32,
                    num_stages=num_stages,
                )
            )

    return configs


# =============================================================================
# XPU-optimized row-wise softmax kernel
# =============================================================================
@triton.autotune(
    configs=_softmax_autotune_configs(),
    key=["M", "N", "stride_xm", "stride_xn", "stride_ym", "stride_yn"],
)
@triton.jit
def _row_softmax_large_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
    LOG2E: tl.constexpr,
    grf_mode: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    neg_inf = -float("inf")

    row64 = row.to(tl.int64)
    row_start_x = x_ptr + row64 * stride_xm
    row_start_y = y_ptr + row64 * stride_ym

    row_max = tl.full((), neg_inf, tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + offs_n
        mask = cols < N
        cols64 = cols.to(tl.int64)
        vals = tl.load(row_start_x + cols64 * stride_xn, mask=mask, other=neg_inf).to(
            tl.float32
        )
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    row_sum = tl.zeros((), tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + offs_n
        mask = cols < N
        cols64 = cols.to(tl.int64)
        vals = tl.load(row_start_x + cols64 * stride_xn, mask=mask, other=neg_inf).to(
            tl.float32
        )
        row_sum += tl.sum(tl.math.exp2((vals - row_max) * LOG2E), axis=0)

    inv_row_sum = 1.0 / row_sum
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + offs_n
        mask = cols < N
        cols64 = cols.to(tl.int64)
        vals = tl.load(row_start_x + cols64 * stride_xn, mask=mask, other=neg_inf).to(
            tl.float32
        )
        probs = tl.math.exp2((vals - row_max) * LOG2E) * inv_row_sum
        tl.store(
            row_start_y + cols64 * stride_yn,
            probs.to(y_ptr.dtype.element_ty),
            mask=mask,
        )


def kernel_function(
    x, w, b, p=0.2, dim=1, training=False, dropout_p=None, softmax_dim=None
):
    if dropout_p is not None:
        p = dropout_p
    if softmax_dim is not None:
        dim = softmax_dim

    assert (
        isinstance(x, torch.Tensor)
        and isinstance(w, torch.Tensor)
        and isinstance(b, torch.Tensor)
    )
    assert x.ndim == 2 and w.ndim == 2 and b.ndim == 1
    assert dim == 1
    assert (
        x.dtype == torch.float16
        and w.dtype == torch.float16
        and b.dtype == torch.float16
    )

    x_xpu = x.to(device="xpu", dtype=torch.float16).contiguous()
    w_xpu = w.to(device="xpu", dtype=torch.float16).contiguous()
    b_xpu = b.to(device="xpu", dtype=torch.float16).contiguous()

    logits = F.linear(x_xpu, w_xpu, b_xpu)

    M, N = logits.shape
    y = torch.empty_like(logits)

    _row_softmax_large_kernel[(M,)](
        logits,
        y,
        M,
        N,
        logits.stride(0),
        logits.stride(1),
        y.stride(0),
        y.stride(1),
        grf_mode="auto",
    )
    return y


class Model(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self._packed_weight = None
        self._packed_bias = None
        self._packed_weight_t = None
        self._cache_key = None

    def _prepare_xpu_params(self):
        weight = self.matmul.weight
        bias = self.matmul.bias

        cache_key = (
            weight.data_ptr(),
            bias.data_ptr(),
            tuple(weight.shape),
            tuple(bias.shape),
            str(weight.device),
            str(bias.device),
            weight.dtype,
            bias.dtype,
        )

        if (
            self._cache_key == cache_key
            and self._packed_weight is not None
            and self._packed_bias is not None
        ):
            return

        with torch.no_grad():
            weight_xpu = (
                weight.detach().to(device="xpu", dtype=torch.float16).contiguous()
            )
            bias_xpu = bias.detach().to(device="xpu", dtype=torch.float16).contiguous()

        self._packed_weight = weight_xpu
        self._packed_bias = bias_xpu
        self._packed_weight_t = weight_xpu.transpose(0, 1).contiguous()
        self._cache_key = cache_key

    def forward(self, x):
        self._prepare_xpu_params()

        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to(device="xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        return kernel_function(
            x,
            self._packed_weight,
            self._packed_bias,
            self.dropout_p,
        )
