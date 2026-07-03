# ruff: noqa: E731
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------------
# Autotune config helpers
# ------------------------------------------------------------------------
def _linear_autotune_configs():
    configs = []

    # Intel XPU-oriented GEMM search space.
    # Keep broad coverage across:
    # - small tiles for fallback / small shapes
    # - medium tiles for balanced occupancy
    # - large 256x* and 256x256 tiles for compute-bound large problems
    # - required 32-warp large-tile configs for XPU
    gemm_tiles = [
        # small / medium region
        (64, 64, 32, 4, 2),
        (64, 64, 64, 8, 2),
        (64, 128, 32, 8, 2),
        (64, 128, 64, 8, 2),
        (128, 64, 32, 8, 2),
        (128, 64, 64, 8, 2),
        (128, 128, 32, 8, 2),
        (128, 128, 64, 16, 2),
        (128, 128, 32, 16, 3),
        # medium / large region
        (64, 256, 32, 16, 2),
        (64, 256, 64, 16, 2),
        (128, 256, 32, 16, 2),
        (128, 256, 64, 16, 2),
        (256, 128, 32, 16, 2),
        (256, 128, 64, 16, 2),
        # XPU-focused large-tile region
        (256, 128, 32, 32, 2),
        (256, 128, 32, 32, 3),
        (256, 128, 64, 32, 2),
        (128, 256, 32, 32, 2),
        (128, 256, 32, 32, 3),
        (128, 256, 64, 32, 2),
        (256, 256, 16, 32, 3),  # recommended XPU config
        (256, 256, 32, 32, 2),
        (256, 256, 32, 32, 3),
        (256, 256, 64, 32, 2),
        (256, 256, 64, 32, 3),
    ]

    seen = set()
    for bm, bn, bk, nw, ns in gemm_tiles:
        group_sizes = (1, 2, 4, 8) if bm < 256 else (1, 2, 4)
        for group_size_m in group_sizes:
            key = (bm, bn, bk, nw, ns, group_size_m)
            if key in seen:
                continue
            seen.add(key)
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                        "GROUP_SIZE_M": group_size_m,
                    },
                    num_warps=nw,
                    num_stages=ns,
                )
            )
    return configs


def _bn_stats_autotune_configs():
    # Separate reduction-style autotune family.
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=32, num_stages=2),
    ]


def _bn_apply_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=32, num_stages=2),
    ]


# ------------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------------
@triton.autotune(configs=_linear_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _linear_fwd_kernel(
    x_ptr,
    wt_ptr,  # logical shape [K, N]
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wtk,
    stride_wtn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_width = GROUP_SIZE_M * num_pid_n
    group_id = pid // group_width
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % group_width
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    wt_bp = tl.make_block_ptr(
        base=wt_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(wt_bp, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        wt_bp = tl.advance(wt_bp, (BLOCK_K, 0))

    offs_n_vec = offs_n + tl.arange(0, BLOCK_N)
    bias = tl.load(b_ptr + offs_n_vec, mask=offs_n_vec < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc, boundary_check=(0, 1))


@triton.autotune(configs=_bn_stats_autotune_configs(), key=["M", "N"])
@triton.jit
def _bn_stats_kernel(
    x_ptr,
    scale_ptr,
    mean_ptr,
    invstd_ptr,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_s,
    stride_mean,
    stride_invstd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_n = tl.program_id(0)
    col_start = pid_n * BLOCK_N
    offs_n = col_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    sum_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq_val = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        x_bp = tl.make_block_ptr(
            base=x_ptr,
            shape=(M, N),
            strides=(stride_xm, stride_xn),
            offsets=(m_start, col_start),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        x = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + offs_n * stride_mean, mean, mask=n_mask)
    tl.store(invstd_ptr + offs_n * stride_invstd, invstd, mask=n_mask)


@triton.autotune(configs=_bn_apply_autotune_configs(), key=["M", "N"])
@triton.jit
def _bn_apply_kernel(
    x_ptr,
    scale_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_s,
    stride_mean,
    stride_invstd,
    stride_gamma,
    stride_beta,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    offs_n_vec = offs_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n_vec < N

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    x = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    s = tl.load(scale_ptr + offs_n_vec * stride_s, mask=n_mask, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + offs_n_vec * stride_mean, mask=n_mask, other=0.0).to(
        tl.float32
    )
    invstd = tl.load(
        invstd_ptr + offs_n_vec * stride_invstd, mask=n_mask, other=0.0
    ).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n_vec * stride_gamma, mask=n_mask, other=0.0).to(
        tl.float32
    )
    beta = tl.load(beta_ptr + offs_n_vec * stride_beta, mask=n_mask, other=0.0).to(
        tl.float32
    )

    var = 1.0 / (invstd * invstd) - eps
    var = tl.maximum(var, 0.0)
    coeff = (s * tl.rsqrt(s * s * var + eps)) * gamma
    out = (x - mean[None, :]) * coeff[None, :] + beta[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, out, boundary_check=(0, 1))


# ------------------------------------------------------------------------
# Top-level wrapper
# ------------------------------------------------------------------------
def _get_hw_num_progs():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            cap = torch.xpu.get_device_capability(0)
            if isinstance(cap, dict):
                for key in (
                    "gpu_subslice_count",
                    "max_compute_units",
                    "subslice_count",
                ):
                    if key in cap:
                        val = int(cap[key])
                        if val > 0:
                            return val
            elif isinstance(cap, (tuple, list)) and len(cap) > 0:
                val = int(cap[0])
                if val > 0:
                    return val
        except Exception:
            pass
    return 1


def kernel_function(x, weight_t, bias, scale, bn_weight, bn_bias, eps):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if weight_t.device.type != "xpu" or weight_t.dtype != torch.float16:
        weight_t_xpu = weight_t.to("xpu", dtype=torch.float16).contiguous()
    else:
        weight_t_xpu = weight_t.contiguous()

    if bias.device.type != "xpu" or bias.dtype != torch.float16:
        bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bias_xpu = bias.contiguous()

    if scale.device.type != "xpu" or scale.dtype != torch.float16:
        scale_xpu = scale.to("xpu", dtype=torch.float16).contiguous()
    else:
        scale_xpu = scale.contiguous()

    if bn_weight.device.type != "xpu" or bn_weight.dtype != torch.float16:
        bn_weight_xpu = bn_weight.to("xpu", dtype=torch.float16).contiguous()
    else:
        bn_weight_xpu = bn_weight.contiguous()

    if bn_bias.device.type != "xpu" or bn_bias.dtype != torch.float16:
        bn_bias_xpu = bn_bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bn_bias_xpu = bn_bias.contiguous()

    M, K = x_xpu.shape
    Kt, N = weight_t_xpu.shape
    if (
        K != Kt
        or bias_xpu.shape[0] != N
        or scale_xpu.shape[0] != N
        or bn_weight_xpu.shape[0] != N
        or bn_bias_xpu.shape[0] != N
    ):
        raise ValueError("Shape mismatch")

    y_lin = torch.empty((M, N), device="xpu", dtype=torch.float32)
    mean = torch.empty((N,), device="xpu", dtype=torch.float32)
    invst = torch.empty((N,), device="xpu", dtype=torch.float32)
    y_out = torch.empty((M, N), device="xpu", dtype=torch.float32)

    s_xm, s_xk = x_xpu.stride(0), x_xpu.stride(1)
    s_wtk, s_wtn = weight_t_xpu.stride(0), weight_t_xpu.stride(1)
    s_ym, s_yn = y_lin.stride(0), y_lin.stride(1)
    s_s = scale_xpu.stride(0)
    s_m = mean.stride(0)
    s_i = invst.stride(0)
    s_g = bn_weight_xpu.stride(0)
    s_bb = bn_bias_xpu.stride(0)
    s_om, s_on = y_out.stride(0), y_out.stride(1)

    _ = _get_hw_num_progs()

    def _grid_linear(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _linear_fwd_kernel[_grid_linear](
        x_xpu,
        weight_t_xpu,
        bias_xpu,
        y_lin,
        M,
        N,
        K,
        s_xm,
        s_xk,
        s_wtk,
        s_wtn,
        s_ym,
        s_yn,
        grf_mode="auto",
    )

    def _grid_stats(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    _bn_stats_kernel[_grid_stats](
        y_lin,
        scale_xpu,
        mean,
        invst,
        M,
        N,
        float(eps),
        s_ym,
        s_yn,
        s_s,
        s_m,
        s_i,
        grf_mode="auto",
    )

    def _grid_apply(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _bn_apply_kernel[_grid_apply](
        y_lin,
        scale_xpu,
        mean,
        invst,
        bn_weight_xpu,
        bn_bias_xpu,
        y_out,
        M,
        N,
        float(eps),
        s_ym,
        s_yn,
        s_s,
        s_m,
        s_i,
        s_g,
        s_bb,
        s_om,
        s_on,
        grf_mode="auto",
    )

    return y_out.to(x.dtype)


# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

        self._packed_weight_xpu = None
        self._packed_bias_xpu = None
        self._scale_xpu = None
        self._bn_weight_xpu = None
        self._bn_bias_xpu = None

        self._packed_weight_meta = None
        self._packed_bias_meta = None
        self._scale_meta = None
        self._bn_weight_meta = None
        self._bn_bias_meta = None

    @staticmethod
    def _tensor_cache_meta(t):
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.device.type, t.dtype)

    def _ensure_xpu_cached_params(self):
        weight = self.gemm.weight
        bias = self.gemm.bias
        scale = self.scale
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias

        weight_meta = self._tensor_cache_meta(weight)
        if self._packed_weight_xpu is None or self._packed_weight_meta != weight_meta:
            weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
            self._packed_weight_xpu = weight_xpu.t().contiguous()
            self._packed_weight_meta = weight_meta

        bias_meta = self._tensor_cache_meta(bias)
        if self._packed_bias_xpu is None or self._packed_bias_meta != bias_meta:
            self._packed_bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
            self._packed_bias_meta = bias_meta

        scale_meta = self._tensor_cache_meta(scale)
        if self._scale_xpu is None or self._scale_meta != scale_meta:
            self._scale_xpu = scale.to("xpu", dtype=torch.float16).contiguous()
            self._scale_meta = scale_meta

        bn_weight_meta = self._tensor_cache_meta(bn_weight)
        if self._bn_weight_xpu is None or self._bn_weight_meta != bn_weight_meta:
            self._bn_weight_xpu = bn_weight.to("xpu", dtype=torch.float16).contiguous()
            self._bn_weight_meta = bn_weight_meta

        bn_bias_meta = self._tensor_cache_meta(bn_bias)
        if self._bn_bias_xpu is None or self._bn_bias_meta != bn_bias_meta:
            self._bn_bias_xpu = bn_bias.to("xpu", dtype=torch.float16).contiguous()
            self._bn_bias_meta = bn_bias_meta

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        self._ensure_xpu_cached_params()

        return kernel_function(
            x,
            self._packed_weight_xpu,
            self._packed_bias_xpu,
            self._scale_xpu,
            self._bn_weight_xpu,
            self._bn_bias_xpu,
            self.bn.eps,
        )


# ------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------
def run_test():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available, skipping test.")
        sys.exit(0)

    in_f, out_f, scale_sh = get_init_inputs()
    model = Model(in_f, out_f, scale_sh).to("xpu")
    model.train()
    x = get_inputs()[0].to("xpu", dtype=torch.float16)

    y_ref = model.bn(model.gemm(x) * model.scale)
    y_pred = model(x)

    if not torch.allclose(y_ref, y_pred, rtol=1e-2, atol=1e-2):
        max_err = (y_ref - y_pred).abs().max().item()
        print(f"FAIL: max error {max_err}")
        sys.exit(1)
    print("PASS")
    sys.exit(0)
