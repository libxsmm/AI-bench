# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _gemm_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 1}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 2}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=2),
    ]


@triton.autotune(
    configs=_gemm_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _fused_linear_bias_hardtanh_mish_kernel(
    x_ptr,
    w_t_ptr,
    fused_bias_ptr,
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
    MIN_VAL: tl.constexpr,
    MAX_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if GROUP_SIZE_M > 1 and num_pid_m > 1:
        group_width = GROUP_SIZE_M * num_pid_n
        group_id = pid // group_width
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_in_group = pid % group_width
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    a_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_bp = tl.make_block_ptr(
        base=w_t_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_bp, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.max_contiguous(offs_n, BLOCK_N)
    fused_bias = tl.load(fused_bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += fused_bias[None, :]

    acc = tl.maximum(tl.minimum(acc, MAX_VAL), MIN_VAL)

    # Reduce register pressure by reusing the accumulator tile for the full Mish epilogue.
    # mish(x) = x * tanh(softplus(x))
    # For x in [-1, 1] after hardtanh:
    # tanh(softplus(x)) = ((1 + exp(x))^2 - 1) / ((1 + exp(x))^2 + 1)
    x_clamped = acc
    log2e = 1.4426950408889634

    acc = tl.math.exp2(acc * log2e)   # exp(x)
    acc = acc + 1.0                   # 1 + exp(x)
    acc = acc * acc                   # (1 + exp(x))^2
    acc = (acc - 1.0) / (acc + 1.0)   # tanh(softplus(x))
    acc = x_clamped * acc             # mish(x)

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_C": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_C": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_C": 64}, num_warps=4, num_stages=1),
    ],
    key=["C", "G", "CHANNELS_PER_GROUP"],
)
@triton.jit
def _groupnorm_affine_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    N,
    C,
    G,
    stride_xn,
    stride_xc,
    stride_yn,
    stride_yc,
    stride_gc,
    stride_bc,
    eps,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK_C)
    c = g * CHANNELS_PER_GROUP + offs
    mask = (n < N) & (offs < CHANNELS_PER_GROUP) & (c < C)

    n64 = n.to(tl.int64)
    c64 = c.to(tl.int64)

    x_ptrs = x_ptr + n64 * stride_xn + c64 * stride_xc
    y_ptrs = y_ptr + n64 * stride_yn + c64 * stride_yc

    x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    gamma_v = tl.load(gamma_ptr + c64 * stride_gc, mask=mask, other=0.0).to(tl.float32)
    beta_v = tl.load(beta_ptr + c64 * stride_bc, mask=mask, other=0.0).to(tl.float32)

    inv_cpg = 1.0 / CHANNELS_PER_GROUP
    sum_x = tl.sum(x_val, axis=0)
    sum_x2 = tl.sum(x_val * x_val, axis=0)
    mean = sum_x * inv_cpg
    var = sum_x2 * inv_cpg - mean * mean
    inv_std = tl.rsqrt(var + eps)

    y_val = (x_val - mean) * inv_std
    y_val = y_val * gamma_v + beta_v
    tl.store(y_ptrs, y_val.to(y_ptr.dtype.element_ty), mask=mask)


def kernel_function(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    gemm_bias: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int = 256,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU is required"

    x_xpu = x.to(device="xpu", dtype=torch.float16).contiguous()
    weight_t_xpu = weight_t.to(device="xpu", dtype=torch.float16).contiguous()
    gamma_xpu = gamma.to(device="xpu", dtype=torch.float16).contiguous()
    beta_xpu = beta.to(device="xpu", dtype=torch.float16).contiguous()
    gemm_bias_xpu = gemm_bias.to(device="xpu", dtype=torch.float16).contiguous()
    bias_xpu = bias.to(device="xpu", dtype=torch.float16).contiguous()

    fused_bias_xpu = (gemm_bias_xpu + bias_xpu).contiguous()

    M, K = x_xpu.shape
    K2, N = weight_t_xpu.shape
    assert K == K2
    assert fused_bias_xpu.numel() == N
    assert gamma_xpu.numel() == N and beta_xpu.numel() == N

    y1 = torch.empty((M, N), device=x_xpu.device, dtype=x_xpu.dtype)

    grid1 = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _fused_linear_bias_hardtanh_mish_kernel[grid1](
        x_xpu,
        weight_t_xpu,
        fused_bias_xpu,
        y1,
        M,
        N,
        K,
        x_xpu.stride(0),
        x_xpu.stride(1),
        weight_t_xpu.stride(0),
        weight_t_xpu.stride(1),
        y1.stride(0),
        y1.stride(1),
        -1.0,
        1.0,
        grf_mode="auto",
    )

    y2 = torch.empty_like(y1)
    G = int(num_groups)
    N2, C = y1.shape
    assert C % G == 0
    channels_per_group = C // G

    grid2 = (N2 * G,)
    _groupnorm_affine_kernel[grid2](
        y1,
        gamma_xpu,
        beta_xpu,
        y2,
        N2,
        C,
        G,
        y1.stride(0),
        y1.stride(1),
        y2.stride(0),
        y2.stride(1),
        gamma_xpu.stride(0),
        beta_xpu.stride(0),
        float(eps),
        CHANNELS_PER_GROUP=channels_per_group,
    )
    return y2


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        self._params_prepared = False
        self._packed_weight_t = None
        self._packed_weight_version = -1
        self._fused_bias_cache = None
        self._fused_bias_versions = (-1, -1)

    def _prepare_xpu_params_once(self):
        if not self._params_prepared:
            if self.gemm.weight.device.type != "xpu" or self.gemm.weight.dtype != torch.float16:
                self.gemm.weight.data = self.gemm.weight.data.to(device="xpu", dtype=torch.float16).contiguous()
            elif not self.gemm.weight.is_contiguous():
                self.gemm.weight.data = self.gemm.weight.data.contiguous()

            if self.gemm.bias.device.type != "xpu" or self.gemm.bias.dtype != torch.float16:
                self.gemm.bias.data = self.gemm.bias.data.to(device="xpu", dtype=torch.float16).contiguous()
            elif not self.gemm.bias.is_contiguous():
                self.gemm.bias.data = self.gemm.bias.data.contiguous()

            if self.bias.device.type != "xpu" or self.bias.dtype != torch.float16:
                self.bias.data = self.bias.data.to(device="xpu", dtype=torch.float16).contiguous()
            elif not self.bias.is_contiguous():
                self.bias.data = self.bias.data.contiguous()

            if self.group_norm.weight.device.type != "xpu" or self.group_norm.weight.dtype != torch.float16:
                self.group_norm.weight.data = self.group_norm.weight.data.to(device="xpu", dtype=torch.float16).contiguous()
            elif not self.group_norm.weight.is_contiguous():
                self.group_norm.weight.data = self.group_norm.weight.data.contiguous()

            if self.group_norm.bias.device.type != "xpu" or self.group_norm.bias.dtype != torch.float16:
                self.group_norm.bias.data = self.group_norm.bias.data.to(device="xpu", dtype=torch.float16).contiguous()
            elif not self.group_norm.bias.is_contiguous():
                self.group_norm.bias.data = self.group_norm.bias.data.contiguous()

            self._params_prepared = True

    def _get_packed_weight_t(self):
        w = self.gemm.weight
        if (
            self._packed_weight_t is None
            or self._packed_weight_version != int(w._version)
            or self._packed_weight_t.device != w.device
        ):
            self._packed_weight_t = w.transpose(0, 1).contiguous()
            self._packed_weight_version = int(w._version)
        return self._packed_weight_t

    def _get_fused_bias(self):
        gb = self.gemm.bias
        b = self.bias
        versions = (int(gb._version), int(b._version))
        if (
            self._fused_bias_cache is None
            or self._fused_bias_versions != versions
            or self._fused_bias_cache.device != gb.device
        ):
            self._fused_bias_cache = (gb + b).contiguous()
            self._fused_bias_versions = versions
        return self._fused_bias_cache

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to(device="xpu", dtype=torch.float16).contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        self._prepare_xpu_params_once()
        packed_weight_t = self._get_packed_weight_t()
        fused_bias = self._get_fused_bias()

        return kernel_function(
            x,
            packed_weight_t,
            fused_bias,
            torch.zeros_like(fused_bias),
            self.group_norm.weight,
            self.group_norm.bias,
            num_groups=self.group_norm.num_groups,
        )
