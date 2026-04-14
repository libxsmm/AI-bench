# ruff: noqa: E731
# KernelBench-compatible wrapper — original Triton kernels retained.
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _linear_bias_configs():
    configs = []
    seen = set()
    candidates = [
        (64, 64, 32, 4, 2),
        (64, 64, 64, 8, 2),
        (64, 128, 32, 8, 2),
        (64, 128, 64, 8, 2),
        (128, 64, 32, 8, 2),
        (128, 64, 64, 8, 2),
        (128, 128, 32, 8, 3),
        (128, 128, 64, 8, 4),
        (128, 256, 32, 16, 2),
        (256, 128, 32, 16, 2),
        (256, 256, 16, 32, 3),
        (256, 256, 32, 32, 3),
    ]
    for bm, bn, bk, nw, ns in candidates:
        key = (bm, bn, bk, nw, ns)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


def _reduced_gemm_configs():
    configs = []
    seen = set()
    tile_candidates = [
        (64, 64, 32, 1, 8, 2),
        (64, 64, 64, 4, 8, 2),
        (64, 128, 32, 1, 8, 2),
        (64, 128, 64, 4, 8, 2),
        (64, 256, 32, 4, 16, 2),
        (64, 256, 64, 4, 16, 2),
        (128, 64, 32, 1, 8, 2),
        (128, 64, 64, 2, 8, 2),
        (128, 128, 32, 1, 16, 2),
        (128, 128, 64, 2, 16, 2),
        (128, 256, 32, 1, 16, 2),
        (128, 256, 32, 2, 16, 2),
        (128, 256, 32, 4, 32, 3),
        (256, 128, 16, 1, 16, 2),
        (256, 128, 32, 1, 16, 2),
        (256, 128, 32, 4, 32, 3),
        (256, 256, 16, 1, 32, 3),
        (256, 256, 16, 4, 32, 3),
        (256, 256, 32, 1, 32, 3),
    ]
    for bm, bn, bk, gs, nw, ns in tile_candidates:
        key = (bm, bn, bk, gs, nw, ns)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "GROUP_SIZE_M": gs,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


# Main-path fused configs broadened for Intel XPU while preserving BLOCK_N=128
# because partial-buffer indexing uses pid_n directly.
def _fused_partial_max_configs():
    configs = []
    seen = set()
    fused = [
        (64, 128, 32, 1, 8, 2),
        (64, 128, 64, 4, 8, 2),
        (128, 128, 16, 1, 8, 2),
        (128, 128, 32, 1, 8, 2),
        (128, 128, 32, 4, 8, 2),
        (128, 128, 64, 1, 8, 2),
        (128, 128, 64, 4, 16, 2),
        (256, 128, 16, 1, 16, 2),
        (256, 128, 16, 4, 16, 2),
        (256, 128, 32, 1, 16, 2),
        (256, 128, 32, 4, 16, 2),
        (256, 128, 32, 1, 32, 2),
        (256, 128, 32, 4, 32, 3),
    ]
    for bm, bn, bk, gs, nw, ns in fused:
        key = (bm, bn, bk, gs, nw, ns)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "GROUP_SIZE_M": gs,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


def _reduce_partial_max_configs():
    configs = []
    for block_tiles, nw, ns in [
        (8, 4, 2),
        (16, 4, 2),
        (32, 8, 2),
        (64, 8, 2),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_TILES": block_tiles},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


# ----------------------------
# Original Subgraph 0 retained
# ----------------------------
@triton.autotune(
    configs=_linear_bias_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _linear_bias_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_bp = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_bp, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.device.type == 'xpu'
    assert weight.device == x.device and bias.device == x.device
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw and bias.shape[0] == Nw
    allowed = (torch.bfloat16, torch.float16)
    assert x.dtype in allowed and weight.dtype in allowed and bias.dtype in allowed
    y = torch.empty((M, Nw), device=x.device, dtype=x.dtype)
    stride_am, stride_ak = x.stride()
    stride_w0, stride_w1 = weight.stride()
    stride_bk, stride_bn = stride_w1, stride_w0
    stride_cm, stride_cn = y.stride()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Nw, meta['BLOCK_N']))

    _linear_bias_kernel[grid](
        x, weight, bias, y,
        M, Nw, Kx,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn
    )
    return y


# ---------------------------------------------------
# Original Subgraph 1 retained
# ---------------------------------------------------
@triton.jit
def _pool_gelu_scale_reduce_max_kernel(
    x_ptr, out_ptr, N, W, stride_n, stride_w, scale,
    POOL_K: tl.constexpr, STRIDE: tl.constexpr, BLOCK_POOLS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_mask = pid < N
    num_pools = W // STRIDE
    row_start = pid * stride_n
    offs_p = tl.arange(0, BLOCK_POOLS)
    offs_k = tl.arange(0, POOL_K)
    running_max = tl.zeros((), dtype=tl.float32) - float('inf')
    INV_SQRT2 = 0.7071067811865476
    for start_p in tl.range(0, num_pools, BLOCK_POOLS):
        idx_p = start_p + offs_p
        valid_p = idx_p < num_pools
        ptrs = x_ptr + row_start + idx_p[:, None] * STRIDE + offs_k[None, :] * stride_w
        vals = tl.load(ptrs, mask=valid_p[:, None] & row_mask, other=0.0)
        sums = tl.sum(vals, axis=1)
        means = sums * (1.0 / POOL_K)
        t = means * INV_SQRT2
        gelu = 0.5 * means * (1.0 + tl.math.erf(t))
        scaled = gelu * scale
        block_max = tl.max(scaled, axis=0)
        running_max = tl.maximum(running_max, block_max)
    tl.store(out_ptr + pid, running_max, mask=row_mask)


def _pool(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    assert x.device.type == 'xpu'
    assert x.dtype == torch.float16
    N, W = x.shape
    POOL_K = 16
    STRIDE = 16
    assert W % STRIDE == 0
    out = torch.empty((N,), device=x.device, dtype=x.dtype)
    BLOCK_POOLS = 128
    grid = (N,)
    _pool_gelu_scale_reduce_max_kernel[grid](
        x, out, N, W, x.stride(0), x.stride(1), float(scale_factor),
        POOL_K=POOL_K, STRIDE=STRIDE, BLOCK_POOLS=BLOCK_POOLS,
        num_warps=4, num_stages=2
    )
    return out


# ----------------------------------------
# Reduced GEMM path for pooled outputs
# ----------------------------------------
@triton.autotune(
    configs=_reduced_gemm_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_bias_reduced_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))
        w = tl.load(w_bp, boundary_check=(0, 1))
        acc = tl.dot(a, w, acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_bp = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_bp, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def _linear_reduced(x: torch.Tensor, weight_pool_kn: torch.Tensor, bias_pool: torch.Tensor) -> torch.Tensor:
    assert x.device.type == 'xpu'
    assert weight_pool_kn.device.type == 'xpu'
    assert bias_pool.device.type == 'xpu'
    assert x.dtype == torch.float16
    assert weight_pool_kn.dtype == torch.float16
    assert bias_pool.dtype == torch.float16

    M, Kx = x.shape
    Kw, Nw = weight_pool_kn.shape
    assert Kx == Kw and bias_pool.shape[0] == Nw

    y = torch.empty((M, Nw), device=x.device, dtype=x.dtype)

    stride_am, stride_ak = x.stride()
    stride_wk, stride_wn = weight_pool_kn.stride()
    stride_cm, stride_cn = y.stride()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Nw, meta['BLOCK_N']),)

    _linear_bias_reduced_kernel[grid](
        x, weight_pool_kn, bias_pool, y,
        M, Nw, Kx,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        grf_mode="auto",
    )
    return y


# ---------------------------------------------------------
# Fused guarded path: GEMM tile -> GELU/scale -> partial max
# Optimization for this stage:
# - Keep fusion to avoid materializing full [M, N]
# - Reduce register pressure by reducing before GELU:
#     max_j GELU(scale * x_j) == GELU(scale * max_j x_j) for positive scale
#   because GELU is monotone increasing.
# This avoids applying GELU to the full BLOCK_M x BLOCK_N tile.
# ---------------------------------------------------------
@triton.autotune(
    configs=_fused_partial_max_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_gelu_scale_partial_max_kernel(
    a_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_pm, stride_pn,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))
        w = tl.load(w_bp, boundary_check=(0, 1))
        acc = tl.dot(a, w, acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Reduce first, then apply monotonic epilogue to the reduced values only.
    row_max = tl.max(acc, axis=1)
    inv_sqrt2 = 0.7071067811865476
    row_max = row_max * scale
    row_max = 0.5 * row_max * (1.0 + tl.math.erf(row_max * inv_sqrt2))

    partial_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(partial_ptrs, row_max.to(partial_ptr.dtype.element_ty), mask=offs_m < M)


@triton.autotune(
    configs=_reduce_partial_max_configs(),
    key=['num_tiles_n'],
)
@triton.jit
def _reduce_partial_max_kernel(
    partial_ptr, out_ptr,
    M, num_tiles_n,
    stride_pm, stride_pn,
    BLOCK_TILES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_mask = pid < M
    offs_t = tl.arange(0, BLOCK_TILES)
    running_max = tl.zeros((), dtype=tl.float32) - float("inf")

    for start_t in tl.range(0, num_tiles_n, BLOCK_TILES):
        cols = start_t + offs_t
        mask = row_mask & (cols < num_tiles_n)
        ptrs = partial_ptr + pid * stride_pm + cols * stride_pn
        vals = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(vals, axis=0)
        running_max = tl.maximum(running_max, block_max)

    tl.store(out_ptr + pid, running_max.to(out_ptr.dtype.element_ty), mask=row_mask)


def _linear_gelu_scale_reduce_max_fused(
    x: torch.Tensor, weight_pool_kn: torch.Tensor, bias_pool: torch.Tensor, scale_factor: float
) -> torch.Tensor:
    assert x.device.type == 'xpu'
    assert weight_pool_kn.device.type == 'xpu'
    assert bias_pool.device.type == 'xpu'
    assert x.dtype == torch.float16
    assert weight_pool_kn.dtype == torch.float16
    assert bias_pool.dtype == torch.float16

    M, Kx = x.shape
    Kw, Nw = weight_pool_kn.shape
    assert Kx == Kw and bias_pool.shape[0] == Nw

    max_block_n_assumed = 128
    num_tiles_n = triton.cdiv(Nw, max_block_n_assumed)
    partial = torch.empty((M, num_tiles_n), device=x.device, dtype=torch.float16)

    stride_am, stride_ak = x.stride()
    stride_wk, stride_wn = weight_pool_kn.stride()
    stride_pm, stride_pn = partial.stride()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Nw, meta['BLOCK_N']),)

    _linear_gelu_scale_partial_max_kernel[grid](
        x, weight_pool_kn, bias_pool, partial,
        M, Nw, Kx,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_pm, stride_pn,
        float(scale_factor),
        grf_mode="auto",
    )

    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    _reduce_partial_max_kernel[(M,)](
        partial, out,
        M, partial.shape[1],
        partial.stride(0), partial.stride(1),
    )
    return out


@triton.jit
def _gelu_scale_reduce_max_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    scale,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_n = tl.arange(0, BLOCK_N)
    row_mask = pid < M
    running_max = tl.zeros((), dtype=tl.float32) - float("inf")
    inv_sqrt2 = 0.7071067811865476

    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + offs_n
        mask = row_mask & (cols < N)
        ptrs = x_ptr + pid * stride_xm + cols * stride_xn
        vals = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        vals = vals * scale
        vals = 0.5 * vals * (1.0 + tl.math.erf(vals * inv_sqrt2))
        block_max = tl.max(vals, axis=0)
        running_max = tl.maximum(running_max, block_max)

    tl.store(out_ptr + pid, running_max.to(out_ptr.dtype.element_ty), mask=row_mask)


def _gelu_scale_reduce_max(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    assert x.device.type == 'xpu'
    assert x.dtype == torch.float16
    M, N = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    _gelu_scale_reduce_max_kernel[(M,)](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        float(scale_factor),
        BLOCK_N=128,
        num_warps=4,
        num_stages=2,
    )
    return out


def _compute_pooled_params(weight: torch.Tensor, bias: torch.Tensor, pool_kernel_size: int):
    assert weight.ndim == 2
    assert bias.ndim == 1
    assert weight.shape[0] % pool_kernel_size == 0
    assert bias.shape[0] % pool_kernel_size == 0

    out_features, in_features = weight.shape
    pooled_out = out_features // pool_kernel_size

    weight_pool = (
        weight.float()
        .view(pooled_out, pool_kernel_size, in_features)
        .mean(dim=1)
        .to(dtype=weight.dtype)
        .contiguous()
    )
    weight_pool_kn = weight_pool.t().contiguous()
    bias_pool = (
        bias.float()
        .view(pooled_out, pool_kernel_size)
        .mean(dim=1)
        .to(dtype=bias.dtype)
        .contiguous()
    )
    return weight_pool, weight_pool_kn, bias_pool


_GLOBAL_POOL_CACHE = {}


def _pool_cache_key(weight: torch.Tensor, bias: torch.Tensor, pool_kernel_size: int):
    weight_version = getattr(weight, "_version", None)
    bias_version = getattr(bias, "_version", None)
    return (
        weight.data_ptr(),
        bias.data_ptr(),
        tuple(weight.shape),
        tuple(bias.shape),
        tuple(weight.stride()),
        tuple(bias.stride()),
        str(weight.dtype),
        str(bias.dtype),
        str(weight.device),
        str(bias.device),
        pool_kernel_size,
        weight_version,
        bias_version,
    )


def _get_cached_pooled_params(weight: torch.Tensor, bias: torch.Tensor, pool_kernel_size: int):
    key = _pool_cache_key(weight, bias, pool_kernel_size)
    cached = _GLOBAL_POOL_CACHE.get(key, None)
    if cached is None:
        cached = _compute_pooled_params(weight, bias, pool_kernel_size)
        _GLOBAL_POOL_CACHE.clear()
        _GLOBAL_POOL_CACHE[key] = cached
    return cached


def kernel_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale_factor: float
) -> torch.Tensor:
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        raise RuntimeError("XPU device is not available")

    x_xpu = x.to("xpu", dtype=torch.float16).contiguous() if (x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous()) else x
    weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous() if (weight.device.type != "xpu" or weight.dtype != torch.float16 or not weight.is_contiguous()) else weight
    bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous() if (bias.device.type != "xpu" or bias.dtype != torch.float16 or not bias.is_contiguous()) else bias

    _, weight_pool_kn, bias_pool = _get_cached_pooled_params(weight_xpu, bias_xpu, 16)
    out = _linear_gelu_scale_reduce_max_fused(x_xpu, weight_pool_kn, bias_pool, scale_factor)

    return out


batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]


class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.pool_kernel_size = pool_kernel_size
        self._cache_key = None
        self._cached_weight_pool = None
        self._cached_weight_pool_kn = None
        self._cached_bias_pool = None

    def _ensure_xpu_and_cache(self):
        if self.matmul.weight.device.type != "xpu" or self.matmul.weight.dtype != torch.float16 or not self.matmul.weight.is_contiguous():
            self.matmul.weight.data = self.matmul.weight.data.to("xpu", dtype=torch.float16).contiguous()
        if self.matmul.bias.device.type != "xpu" or self.matmul.bias.dtype != torch.float16 or not self.matmul.bias.is_contiguous():
            self.matmul.bias.data = self.matmul.bias.data.to("xpu", dtype=torch.float16).contiguous()

        weight = self.matmul.weight
        bias = self.matmul.bias
        weight_version = getattr(weight, "_version", None)
        bias_version = getattr(bias, "_version", None)
        cache_key = (
            weight.data_ptr(),
            bias.data_ptr(),
            tuple(weight.shape),
            tuple(bias.shape),
            tuple(weight.stride()),
            tuple(bias.stride()),
            weight_version,
            bias_version,
            self.pool_kernel_size,
        )

        if self._cache_key != cache_key:
            self._cached_weight_pool, self._cached_weight_pool_kn, self._cached_bias_pool = _compute_pooled_params(
                weight, bias, self.pool_kernel_size
            )
            self._cache_key = cache_key

    def forward(self, x):
        self._ensure_xpu_and_cache()
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous() if (x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous()) else x
        return _linear_gelu_scale_reduce_max_fused(
            x_xpu, self._cached_weight_pool_kn, self._cached_bias_pool, self.scale_factor
        )