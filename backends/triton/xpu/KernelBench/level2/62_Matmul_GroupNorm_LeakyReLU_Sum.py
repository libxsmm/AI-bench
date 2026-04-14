# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Kept for reference only; not used in the execution path.
# Per stage guidance, we avoid this heavily fused design because it creates
# excessive register pressure on Intel XPU for the current workload.
@triton.jit
def _fused_linear_gn_leaky_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_ym, stride_yn,
    eps, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k0 = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k0
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wk

        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    valid_n = tl.sum(mask_n.to(tl.int32), axis=0)
    valid_n_f = valid_n.to(tl.float32)
    mean = tl.sum(tl.where(mask_n[None, :], acc, 0.0), axis=1) / valid_n_f
    centered = tl.where(mask_n[None, :], acc - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / valid_n_f
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    norm = centered * inv_std[:, None]
    out = norm * gamma[None, :] + beta[None, :]
    out = tl.where(out >= 0, out, out * negative_slope)
    out = out * 2.0

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _add_2d_kernel(
    x_ptr, y_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(offs_n, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    x_tile = tl.load(x_ptrs, mask=mask, other=0.0)
    y_tile = tl.load(y_ptrs, mask=mask, other=0.0)
    tl.store(o_ptrs, x_tile + y_tile, mask=mask)


@triton.jit
def _groupnorm_leaky_scale2_kernel_grouped(
    x_ptr,           # [M, N]
    gamma_ptr,       # [N]
    beta_ptr,        # [N]
    y_ptr,           # [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps, negative_slope,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROW_GROUP: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_g = tl.cdiv(N, GROUP_SIZE)

    pids_per_group = ROW_GROUP * num_pid_g
    group_id = pid // pids_per_group
    first_pid_m = group_id * ROW_GROUP

    pid_in_group = pid % pids_per_group
    pid_g = pid_in_group % num_pid_g
    pid_m = first_pid_m + (pid_in_group // num_pid_g)

    # Guard out-of-range grouped rows explicitly to avoid useless memory traffic
    # from overprovisioned programs in the final launch group.
    if pid_m >= num_pid_m:
        return

    row_start = pid_m * BLOCK_M
    col_start = pid_g * GROUP_SIZE

    offs_n = col_start + tl.arange(0, GROUP_SIZE)
    mask_n = offs_n < N
    offs_n = tl.max_contiguous(offs_n, GROUP_SIZE)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, GROUP_SIZE),
        order=(1, 0),
    )
    x = tl.load(x_bp, boundary_check=(0, 1)).to(tl.float32)

    inv_group = 1.0 / GROUP_SIZE
    mean = tl.sum(x, axis=1) * inv_group
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) * inv_group
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    y = x_centered * inv_std[:, None]
    y = y * gamma[None, :] + beta[None, :]
    y = tl.where(y >= 0, y, y * negative_slope)
    y = y * 2.0

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, GROUP_SIZE),
        order=(1, 0),
    )
    tl.store(y_bp, y.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _groupnorm_leaky_scale2_kernel_grouped_noboundary(
    x_ptr,           # [M, N]
    gamma_ptr,       # [N]
    beta_ptr,        # [N]
    y_ptr,           # [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps, negative_slope,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROW_GROUP: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_g = tl.cdiv(N, GROUP_SIZE)

    pids_per_group = ROW_GROUP * num_pid_g
    group_id = pid // pids_per_group
    first_pid_m = group_id * ROW_GROUP

    pid_in_group = pid % pids_per_group
    pid_g = pid_in_group % num_pid_g
    pid_m = first_pid_m + (pid_in_group // num_pid_g)

    if pid_m >= num_pid_m:
        return

    row_start = pid_m * BLOCK_M
    col_start = pid_g * GROUP_SIZE

    offs_n = col_start + tl.arange(0, GROUP_SIZE)
    offs_n = tl.max_contiguous(offs_n, GROUP_SIZE)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, GROUP_SIZE),
        order=(1, 0),
    )
    x = tl.load(x_bp).to(tl.float32)

    inv_group = 1.0 / GROUP_SIZE
    mean = tl.sum(x, axis=1) * inv_group
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) * inv_group
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_n).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n).to(tl.float32)

    y = x_centered * inv_std[:, None]
    y = y * gamma[None, :] + beta[None, :]
    y = tl.where(y >= 0, y, y * negative_slope)
    y = y * 2.0

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, GROUP_SIZE),
        order=(1, 0),
    )
    tl.store(y_bp, y.to(y_ptr.dtype.element_ty))


def kernel_function(
    x: torch.Tensor,
    fc_weight: torch.Tensor,
    fc_bias: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
    eps: float,
    negative_slope: float,
) -> torch.Tensor:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU is not available.")

    x_xpu = x.to(device="xpu", dtype=torch.float16).contiguous()
    w_xpu = fc_weight.to(device="xpu", dtype=torch.float16).contiguous()
    b_xpu = fc_bias.to(device="xpu", dtype=torch.float16).contiguous()
    gw_xpu = gn_weight.to(device="xpu", dtype=torch.float16).contiguous()
    gb_xpu = gn_bias.to(device="xpu", dtype=torch.float16).contiguous()

    if x_xpu.dim() != 2:
        raise ValueError("x must be [M, K]")
    M, K = x_xpu.shape

    if w_xpu.dim() != 2:
        raise ValueError("fc_weight must be [N, K]")
    N, Kw = w_xpu.shape
    if Kw != K:
        raise ValueError("Incompatible fc_weight shape.")
    if b_xpu.numel() != N:
        raise ValueError("fc_bias length != N")
    if gw_xpu.numel() != N or gb_xpu.numel() != N:
        raise ValueError("gn_weight/gn_bias length != N")
    if N % num_groups != 0:
        raise ValueError("N must be divisible by num_groups.")

    group_size = N // num_groups
    if group_size <= 0:
        raise ValueError("Invalid group size.")

    # Per stage guidance: keep vendor GEMM and optimize the lighter epilogue kernel.
    lin = F.linear(x_xpu, w_xpu, b_xpu)
    out = torch.empty_like(lin)

    stride_xm, stride_xn = lin.stride()
    stride_ym, stride_yn = out.stride()

    BLOCK_M = 256
    ROW_GROUP = 4
    num_pid_m = triton.cdiv(M, BLOCK_M)
    total_programs = num_pid_m * num_groups

    # Fast specialized path for this benchmark shape:
    # - M=1024 divisible by BLOCK_M=256
    # - N divisible by GROUP_SIZE
    # This removes boundary checks and reduces address/control overhead.
    if (M % BLOCK_M == 0) and (N % group_size == 0):
        _groupnorm_leaky_scale2_kernel_grouped_noboundary[(total_programs,)](
            lin, gw_xpu, gb_xpu, out,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            eps, negative_slope,
            GROUP_SIZE=group_size,
            BLOCK_M=BLOCK_M,
            ROW_GROUP=ROW_GROUP,
            num_warps=8,
            num_stages=3,
        )
    else:
        _groupnorm_leaky_scale2_kernel_grouped[(total_programs,)](
            lin, gw_xpu, gb_xpu, out,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            eps, negative_slope,
            GROUP_SIZE=group_size,
            BLOCK_M=BLOCK_M,
            ROW_GROUP=ROW_GROUP,
            num_warps=8,
            num_stages=3,
        )
    return out


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_groups]


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.group_norm = nn.GroupNorm(num_groups, hidden_size, eps=eps)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.negative_slope = negative_slope

        self._linear_weight_xpu = None
        self._linear_bias_xpu = None
        self._gn_weight_xpu = None
        self._gn_bias_xpu = None
        self._linear_weight_version = -1
        self._linear_bias_version = -1
        self._gn_weight_version = -1
        self._gn_bias_version = -1

    def _ensure_xpu_params(self):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("Intel XPU is not available.")

        lw_ver = int(self.linear.weight._version)
        lb_ver = int(self.linear.bias._version)
        gw_ver = int(self.group_norm.weight._version)
        gb_ver = int(self.group_norm.bias._version)

        if self._linear_weight_xpu is None or self._linear_weight_version != lw_ver:
            self._linear_weight_xpu = self.linear.weight.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._linear_weight_version = lw_ver

        if self._linear_bias_xpu is None or self._linear_bias_version != lb_ver:
            self._linear_bias_xpu = self.linear.bias.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._linear_bias_version = lb_ver

        if self._gn_weight_xpu is None or self._gn_weight_version != gw_ver:
            self._gn_weight_xpu = self.group_norm.weight.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._gn_weight_version = gw_ver

        if self._gn_bias_xpu is None or self._gn_bias_version != gb_ver:
            self._gn_bias_xpu = self.group_norm.bias.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._gn_bias_version = gb_ver

    def forward(self, x):
        self._ensure_xpu_params()
        return kernel_function(
            x,
            self._linear_weight_xpu,
            self._linear_bias_xpu,
            self._gn_weight_xpu,
            self._gn_bias_xpu,
            self.group_norm.num_groups,
            self.group_norm.eps,
            self.negative_slope,
        )
