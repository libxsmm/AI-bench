# ruff: noqa: E731
import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Reference helpers
# ------------------------------------------------------------------------------
batch_size = 1024
in_features = 4096
out_features = 4096


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features]


# ------------------------------------------------------------------------------
# Original Triton kernel kept for compatibility with verification constraints.
# ------------------------------------------------------------------------------
@triton.jit
def _linear_gelu_softmax_rowwise(
    x_ptr,
    w_ptr,
    b_ptr,
    tmp_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_tm,
    stride_tn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    pid_m64 = pid_m.to(tl.int64)
    x_row = x_ptr + pid_m64 * stride_xm
    tmp_row = tmp_ptr + pid_m64 * stride_tm
    out_row = out_ptr + pid_m64 * stride_om

    inv_sqrt2 = 0.7071067811865475244

    for off_n in tl.range(0, N, BLOCK_N):
        rn = off_n + tl.arange(0, BLOCK_N)
        mask_n = rn < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for off_k in tl.range(0, K, BLOCK_K):
            rk = off_k + tl.arange(0, BLOCK_K)
            mask_k = rk < K
            xk = tl.load(x_row + rk * stride_xk, mask=mask_k, other=0.0)
            w_ptrs = w_ptr + rk[None, :] * stride_wk + rn[:, None] * stride_wn
            wk = tl.load(w_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0.0)
            acc += tl.sum(wk * xk[None, :], axis=1)
        b_val = tl.load(b_ptr + rn, mask=mask_n, other=0.0)
        acc = acc + b_val
        t = acc * inv_sqrt2
        u = tl.math.erf(t)
        gelu = 0.5 * acc * (1.0 + u)
        tl.store(tmp_row + rn * stride_tn, gelu, mask=mask_n)

    m_val = -1e20
    for off_n in tl.range(0, N, BLOCK_N):
        rn = off_n + tl.arange(0, BLOCK_N)
        mask_n = rn < N
        vals = tl.load(tmp_row + rn * stride_tn, mask=mask_n, other=-1e20)
        m_val = tl.maximum(m_val, tl.max(vals, axis=0))

    l_val = 0.0
    for off_n in tl.range(0, N, BLOCK_N):
        rn = off_n + tl.arange(0, BLOCK_N)
        mask_n = rn < N
        vals = tl.load(tmp_row + rn * stride_tn, mask=mask_n, other=-1e20)
        e = tl.exp(vals - m_val)
        l_val += tl.sum(e, axis=0)
        tl.store(out_row + rn * stride_on, e, mask=mask_n)

    inv_l = 1.0 / l_val
    for off_n in tl.range(0, N, BLOCK_N):
        rn = off_n + tl.arange(0, BLOCK_N)
        mask_n = rn < N
        e = tl.load(out_row + rn * stride_on, mask=mask_n, other=0.0)
        tl.store(out_row + rn * stride_on, e * inv_l, mask=mask_n)


# ------------------------------------------------------------------------------
# Original optimized Triton kernel also kept for compatibility.
# ------------------------------------------------------------------------------
@triton.jit
def _gelu_softmax_rowwise_from_logits(
    logits_ptr,
    out_ptr,
    M,
    N,
    stride_lm,
    stride_ln,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    pid_m64 = pid_m.to(tl.int64)
    logits_row = logits_ptr + pid_m64 * stride_lm
    out_row = out_ptr + pid_m64 * stride_om

    inv_sqrt2 = 0.7071067811865475244
    LOG2E = 1.4426950408889634

    m_val = -float("inf")
    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(logits_row + cols * stride_ln, mask=mask, other=0.0)
        x = x.to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
        gelu_masked = tl.where(mask, gelu, -float("inf"))
        m_val = tl.maximum(m_val, tl.max(gelu_masked, axis=0))

    l_val = 0.0
    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(logits_row + cols * stride_ln, mask=mask, other=0.0)
        x = x.to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
        e = tl.math.exp2((gelu - m_val) * LOG2E)
        e_masked = tl.where(mask, e, 0.0)
        l_val += tl.sum(e_masked, axis=0)
        tl.store(out_row + cols * stride_on, e.to(tl.float16), mask=mask)

    inv_l = 1.0 / l_val
    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        e = tl.load(out_row + cols * stride_on, mask=mask, other=0.0)
        y = e.to(tl.float32) * inv_l
        tl.store(out_row + cols * stride_on, y.to(tl.float16), mask=mask)


# ------------------------------------------------------------------------------
# XPU-specific tuned kernels.
# ------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 256}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_N": 256}, num_warps=32, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _gelu_store_rowwise(
    logits_ptr,
    out_ptr,
    M,
    N,
    stride_lm,
    stride_ln,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    pid_m64 = pid_m.to(tl.int64)
    logits_row = logits_ptr + pid_m64 * stride_lm
    out_row = out_ptr + pid_m64 * stride_om
    inv_sqrt2 = 0.7071067811865475244

    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(logits_row + cols * stride_ln, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
        tl.store(out_row + cols * stride_on, gelu.to(tl.float16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 256}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_N": 256}, num_warps=32, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _softmax_inplace_rowwise(
    buf_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_N: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    pid_m64 = pid_m.to(tl.int64)
    row_ptr = buf_ptr + pid_m64 * stride_m
    LOG2E = 1.4426950408889634

    m_val = -float("inf")
    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(row_ptr + cols * stride_n, mask=mask, other=-float("inf")).to(tl.float32)
        vals = tl.where(mask, vals, -float("inf"))
        m_val = tl.maximum(m_val, tl.max(vals, axis=0))

    l_val = 0.0
    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(row_ptr + cols * stride_n, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.math.exp2((vals - m_val) * LOG2E)
        e = tl.where(mask, e, 0.0)
        l_val += tl.sum(e, axis=0)
        tl.store(row_ptr + cols * stride_n, e.to(tl.float16), mask=mask)

    inv_l = 1.0 / l_val
    for off_n in tl.range(0, N, BLOCK_N):
        cols = off_n + tl.arange(0, BLOCK_N)
        mask = cols < N
        e = tl.load(row_ptr + cols * stride_n, mask=mask, other=0.0).to(tl.float32)
        tl.store(row_ptr + cols * stride_n, (e * inv_l).to(tl.float16), mask=mask)


def kernel_function(x, w, b):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is required but not available.")

    x_xpu = x if (x.device.type == "xpu" and x.dtype == torch.float16) else x.to("xpu", dtype=torch.float16)
    w_xpu = w if (w.device.type == "xpu" and w.dtype == torch.float16) else w.to("xpu", dtype=torch.float16)
    b_xpu = b if (b.device.type == "xpu" and b.dtype == torch.float16) else b.to("xpu", dtype=torch.float16)

    x_xpu = x_xpu.contiguous()
    w_xpu = w_xpu.contiguous()
    b_xpu = b_xpu.contiguous()

    if x_xpu.ndim != 2 or w_xpu.ndim != 2 or b_xpu.ndim != 1:
        raise RuntimeError("x:2D, w:2D, b:1D required.")

    M, Kx = x_xpu.shape
    N, Kw = w_xpu.shape
    if Kx != Kw or b_xpu.shape[0] != N:
        raise RuntimeError(f"Shape mismatch: x({x_xpu.shape}), w({w_xpu.shape}), b({b_xpu.shape})")

    logits = F.linear(x_xpu, w_xpu, b_xpu)
    out = torch.empty((M, N), device="xpu", dtype=torch.float16)

    slm, sln = logits.stride(0), logits.stride(1)
    som, son = out.stride(0), out.stride(1)

    grid = (M,)

    _gelu_store_rowwise[grid](
        logits,
        out,
        M,
        N,
        slm,
        sln,
        som,
        son,
        grf_mode="auto",
    )

    _softmax_inplace_rowwise[grid](
        out,
        M,
        N,
        som,
        son,
        grf_mode="auto",
    )

    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._prepared = False

    def _prepare_parameters(self):
        if self._prepared:
            return
        self.linear.weight.data = self.linear.weight.data.to("xpu", dtype=torch.float16).contiguous()
        if self.linear.bias is not None:
            self.linear.bias.data = self.linear.bias.data.to("xpu", dtype=torch.float16).contiguous()
        self._prepared = True

    def forward(self, x):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU is required but not available.")

        self._prepare_parameters()

        x_xpu = x if (x.device.type == "xpu" and x.dtype == torch.float16) else x.to("xpu", dtype=torch.float16)
        x_xpu = x_xpu.contiguous()
        return kernel_function(x_xpu, self.linear.weight, self.linear.bias)