# ruff: noqa: E731
# KernelBench-compatible wrapper — Model class injected by codegen
# The Triton kernel logic is unchanged from the original source.
import torch
import torch.nn as nn
import triton
import triton.language as tl


# GEMM + bias add + subtract kernel
@triton.jit
def kernel_gemm_subtract(
    x_ptr,  # pointer to input X [B, Fin]
    w_ptr,  # pointer to weight W [Fout, Fin]
    bias_ptr,  # pointer to bias [Fout]
    sub_ptr,  # pointer to subtract [Fout]
    y_ptr,  # pointer to output Y [B, Fout]
    B,
    Fin,
    Fout,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_b,
    stride_s,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.cast(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.int32)
    offs_n = tl.cast(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, Fin, BLOCK_K):
        offs_k = tl.cast(k + tl.arange(0, BLOCK_K), tl.int32)

        # A block: X[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
        mask_a = (offs_m[:, None] < B) & (offs_k[None, :] < Fin)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # B block from W[Fout, Fin], but indexed as W[n, k]
        # Produces [BLOCK_K, BLOCK_N] for tl.dot(a, b)
        b_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_b = (offs_k[:, None] < Fin) & (offs_n[None, :] < Fout)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a, b, acc)

    bias = tl.load(bias_ptr + offs_n * stride_b, mask=offs_n < Fout, other=0.0)
    subv = tl.load(sub_ptr + offs_n * stride_s, mask=offs_n < Fout, other=0.0)

    acc = acc + bias[None, :]
    acc = acc - subv[None, :]

    out_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_out = (offs_m[:, None] < B) & (offs_n[None, :] < Fout)
    tl.store(out_ptrs, acc, mask=mask_out)


# Row-wise mean kernel
@triton.jit
def kernel_row_mean(
    y_ptr,  # input Y [B, F]
    mean_ptr,  # output mean [B]
    B,
    F,
    stride_ym,
    stride_yn,
    stride_mm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.cast(pid * BLOCK_M + tl.arange(0, BLOCK_M), tl.int32)

    sum_val = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for f in range(0, F, BLOCK_N):
        offs_n = tl.cast(f + tl.arange(0, BLOCK_N), tl.int32)

        ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        mask = (offs_m[:, None] < B) & (offs_n[None, :] < F)
        y = tl.load(ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(y, axis=1)

    mean = sum_val / F

    mask_m = offs_m < B
    tl.store(mean_ptr + offs_m * stride_mm, mean, mask=mask_m)


# GELU on a vector
@triton.jit
def kernel_gelu_vector(
    mean_ptr,  # input mean [B]
    gelu_ptr,  # output gelu [B]
    B,
    stride_mm,
    stride_gm,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.cast(pid * BLOCK + tl.arange(0, BLOCK), tl.int32)

    mask = offs < B
    x = tl.load(mean_ptr + offs * stride_mm, mask=mask, other=0.0)

    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(gelu_ptr + offs * stride_gm, y, mask=mask)


# Broadcast add: original X + gelu scalar per row
@triton.jit
def kernel_bcast_add(
    orig_ptr,  # original X [B, F]
    gelu_ptr,  # gelu vector [B]
    out_ptr,  # output [B, F]
    B,
    F,
    stride_om,
    stride_on,
    stride_gm,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.cast(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.int32)
    offs_n = tl.cast(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), tl.int32)

    ptrs_orig = orig_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < B) & (offs_n[None, :] < F)
    orig = tl.load(ptrs_orig, mask=mask, other=0.0)

    h = tl.load(gelu_ptr + offs_m * stride_gm, mask=offs_m < B, other=0.0)

    res = orig + h[:, None]

    ptrs_out = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(ptrs_out, res, mask=mask)


def kernel_function(in_features, out_features, x):
    """
    Triton implementation of:
      original_x = x.clone()
      y = x @ W.T + bias
      y = y - subtract
      mean = mean(y, dim=1)
      gelu_vec = gelu(mean)
      out = original_x + gelu_vec[:, None]
    """
    dev_type = x.device.type
    assert dev_type in ("cuda", "xpu"), f"Input must be on CUDA/XPU, got {dev_type}"

    B, Fin = x.shape
    Fout = out_features
    device = x.device

    assert Fin == in_features, f"Expected in_features={in_features}, got {Fin}"
    assert x.dtype == torch.float16, "This kernel expects float32 input"

    # Simulated parameters
    W = torch.randn(Fout, Fin, device=device, dtype=torch.float16)
    bias = torch.randn(Fout, device=device, dtype=torch.float16)
    subtract = torch.randn(Fout, device=device, dtype=torch.float16)

    # Buffers
    y = torch.empty((B, Fout), device=device, dtype=torch.float16)
    mean = torch.empty((B,), device=device, dtype=torch.float32)
    gelu_vec = torch.empty((B,), device=device, dtype=torch.float16)
    out = torch.empty((B, Fin), device=device, dtype=torch.float16)
    orig = x.contiguous()

    # GEMM + bias - subtract
    META1 = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
    grid1 = (triton.cdiv(B, META1["BLOCK_M"]), triton.cdiv(Fout, META1["BLOCK_N"]))
    kernel_gemm_subtract[grid1](
        x,
        W,
        bias,
        subtract,
        y,
        B,
        Fin,
        Fout,
        x.stride(0),
        x.stride(1),
        W.stride(1),
        W.stride(0),
        bias.stride(0),
        subtract.stride(0),
        y.stride(0),
        y.stride(1),
        **META1,
    )

    # Row mean
    META2 = {"BLOCK_M": 256, "BLOCK_N": 128}
    grid2 = (triton.cdiv(B, META2["BLOCK_M"]),)
    kernel_row_mean[grid2](
        y, mean, B, Fout, y.stride(0), y.stride(1), mean.stride(0), **META2
    )

    # GELU on mean vector
    META3 = {"BLOCK": 256}
    grid3 = (triton.cdiv(B, META3["BLOCK"]),)
    kernel_gelu_vector[grid3](
        mean, gelu_vec, B, mean.stride(0), gelu_vec.stride(0), **META3
    )

    # Broadcast add back to original x
    META4 = {"BLOCK_M": 64, "BLOCK_N": 64}
    grid4 = (triton.cdiv(B, META4["BLOCK_M"]), triton.cdiv(Fin, META4["BLOCK_N"]))
    kernel_bcast_add[grid4](
        orig,
        gelu_vec,
        out,
        B,
        Fin,
        orig.stride(0),
        orig.stride(1),
        gelu_vec.stride(0),
        out.stride(0),
        out.stride(1),
        **META4,
    )

    return out


batch_size = 2048
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x):
        return kernel_function(self.in_features, self.out_features, x)
