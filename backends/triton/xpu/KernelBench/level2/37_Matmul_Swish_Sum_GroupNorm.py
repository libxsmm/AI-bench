# ruff: noqa: E731
import torch
import triton
import triton.language as tl
import torch.nn as nn


# -----------------------------------------------------------------------------
# Reference PyTorch Model Definition (for testing)
# -----------------------------------------------------------------------------
class Model(torch.nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation,
    sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.group_norm = torch.nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = self.matmul(x)
        x = torch.sigmoid(x) * x
        x = x + self.bias
        x = self.group_norm(x)
        return x


batch_size = 8192
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]


# -----------------------------------------------------------------------------
# Subgraph sg0: original Triton GEMM+epilogue kernel (kept as required)
# -----------------------------------------------------------------------------
def _autotune_configs():
    configs = []
    for bm, bn, bk, gsz, nw, ns in [
        (64, 64, 32, 1, 4, 2),
        (64, 64, 64, 1, 8, 2),
        (64, 128, 32, 1, 8, 2),
        (64, 128, 64, 1, 8, 2),
        (64, 256, 32, 8, 16, 2),
        (64, 256, 64, 8, 16, 2),
        (128, 64, 32, 1, 8, 2),
        (128, 64, 64, 1, 8, 2),
        (128, 128, 32, 1, 16, 2),
        (128, 128, 32, 8, 16, 2),
        (128, 128, 64, 8, 16, 2),
        (128, 256, 32, 8, 16, 2),
        (128, 256, 64, 8, 16, 2),
        (256, 128, 32, 8, 16, 3),
        (256, 128, 64, 8, 16, 3),
        (128, 128, 32, 8, 16, 3),
        (128, 256, 32, 8, 16, 3),
        (256, 128, 32, 8, 16, 3),
        (256, 256, 16, 16, 32, 3),
        (256, 256, 32, 16, 32, 3),
        (256, 256, 32, 1, 32, 3),
        (256, 256, 64, 8, 32, 2),
        # extra XPU-oriented coverage
        (128, 256, 32, 1, 16, 3),
        (256, 128, 32, 1, 16, 3),
        (256, 256, 16, 1, 32, 3),
        (256, 256, 16, 8, 32, 3),
        (256, 256, 32, 8, 32, 3),
        (256, 128, 64, 1, 16, 3),
        (128, 256, 64, 1, 16, 3),
    ]:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "GROUP_SIZE_M": gsz,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


@triton.autotune(configs=_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _fused_linear_swish_add_kernel(
    x_ptr, w_ptr, b1_ptr, b2_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    HAS_BIAS1: tl.constexpr, HAS_BIAS2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOG2E: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_SIZE_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_SIZE_M
    group_m_size = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % group_m_size)
    pid_n = pid_in_group // group_m_size

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
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
    k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(k_tiles):
        a = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    if HAS_BIAS1:
        b1 = tl.load(b1_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + b1[None, :]

    sig = 1.0 / (1.0 + tl.math.exp2(-acc * LOG2E))
    acc = acc * sig

    if HAS_BIAS2:
        b2 = tl.load(b2_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + b2[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


def _sg0_fused(
    x: torch.Tensor,
    W_t: torch.Tensor,
    b_linear: torch.Tensor,
    b_add: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 2 and W_t.ndim == 2
    M, Kx = x.shape
    Kw, N = W_t.shape
    assert Kx == Kw and N == b_linear.shape[0] and N == b_add.shape[0]
    assert x.device.type == "xpu"
    for t in (W_t, b_linear, b_add):
        assert t.device == x.device and t.dtype == x.dtype

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    stride_xm, stride_xk = x.stride()
    stride_wk, stride_wn = W_t.stride()
    stride_ym, stride_yn = y.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _fused_linear_swish_add_kernel[grid](
        x, W_t, b_linear, b_add, y,
        M, N, Kx,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        HAS_BIAS1=True, HAS_BIAS2=True,
        LOG2E=1.4426950408889634,
        grf_mode="auto",
    )
    return y


# -----------------------------------------------------------------------------
# Subgraph sg1: original GroupNorm kernel (kept as required)
# -----------------------------------------------------------------------------
def _groupnorm_autotune_configs():
    return [
        triton.Config({"BLOCK_ROWS": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 1}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 4}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=32, num_stages=3),
    ]


@triton.autotune(
    configs=_groupnorm_autotune_configs(),
    key=["N", "C"],
)
@triton.jit
def _groupnorm_affine_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    N, C, stride_n, stride_c, eps,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    row_offsets = pid_n * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c_start = pid_g * CHANNELS_PER_GROUP
    offs = tl.arange(0, CHANNELS_PER_GROUP)
    c_idxs = c_start + offs

    row_mask = row_offsets < N
    col_mask = c_idxs < C
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = x_ptr + row_offsets[:, None].to(tl.int64) * stride_n + c_idxs[None, :].to(tl.int64) * stride_c
    y_ptrs = y_ptr + row_offsets[:, None].to(tl.int64) * stride_n + c_idxs[None, :].to(tl.int64) * stride_c

    x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x_val, axis=1) / float(CHANNELS_PER_GROUP)
    diff = x_val - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / float(CHANNELS_PER_GROUP)
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(weight_ptr + c_idxs, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(bias_ptr + c_idxs, mask=col_mask, other=0.0).to(tl.float32)
    y_val = (x_val - mean[:, None]) * rstd[:, None]
    y_val = y_val * gamma[None, :] + beta[None, :]
    tl.store(y_ptrs, y_val.to(y_ptr.dtype.element_ty), mask=mask)


def _sg1_groupnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5
) -> torch.Tensor:
    assert x.ndim == 2
    N, C = x.shape
    assert weight.ndim == 1 and bias.ndim == 1
    assert weight.numel() == C and bias.numel() == C
    assert x.device.type == "xpu"
    for t in (weight, bias):
        assert t.device == x.device and t.dtype == x.dtype
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    y = torch.empty_like(x)
    stride_n, stride_c = x.stride()

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_ROWS"]), num_groups)

    _groupnorm_affine_kernel[grid](
        x, weight, bias, y,
        N, C, stride_n, stride_c, eps,
        CHANNELS_PER_GROUP=channels_per_group,
        grf_mode="auto",
    )
    return y


# -----------------------------------------------------------------------------
# Fused post-GEMM kernel
# -----------------------------------------------------------------------------
def _post_gemm_autotune_configs():
    return [
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_GROUPS": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 4, "BLOCK_GROUPS": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_GROUPS": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_GROUPS": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 8, "BLOCK_GROUPS": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_GROUPS": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_GROUPS": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_GROUPS": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ROWS": 16, "BLOCK_GROUPS": 8}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_GROUPS": 1}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_GROUPS": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_GROUPS": 4}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 32, "BLOCK_GROUPS": 8}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "BLOCK_GROUPS": 1}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_ROWS": 64, "BLOCK_GROUPS": 2}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "BLOCK_GROUPS": 4}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "BLOCK_GROUPS": 8}, num_warps=32, num_stages=3),
    ]


@triton.autotune(
    configs=_post_gemm_autotune_configs(),
    key=["N_ROWS", "C", "NUM_GROUPS"],
)
@triton.jit
def _swish_bias_groupnorm_kernel(
    x_ptr, b2_ptr, gamma_ptr, beta_ptr, y_ptr,
    N_ROWS, C,
    stride_xn, stride_xc,
    stride_yn, stride_yc,
    eps,
    NUM_GROUPS,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    LOG2E: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_gb = tl.program_id(1)

    row_offsets = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rg_offsets = tl.arange(0, BLOCK_GROUPS * CHANNELS_PER_GROUP)

    group_offsets = pid_gb * BLOCK_GROUPS + (rg_offsets // CHANNELS_PER_GROUP)
    c_in_group = rg_offsets % CHANNELS_PER_GROUP
    c_idx = group_offsets * CHANNELS_PER_GROUP + c_in_group

    row_mask = row_offsets < N_ROWS
    col_mask = (group_offsets < NUM_GROUPS) & (c_idx < C)
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = (
        x_ptr
        + row_offsets[:, None].to(tl.int64) * stride_xn
        + c_idx[None, :].to(tl.int64) * stride_xc
    )
    y_ptrs = (
        y_ptr
        + row_offsets[:, None].to(tl.int64) * stride_yn
        + c_idx[None, :].to(tl.int64) * stride_yc
    )

    v = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    v_sig = 1.0 / (1.0 + tl.math.exp2(-v * LOG2E))
    v = v * v_sig

    b2 = tl.load(b2_ptr + c_idx, mask=col_mask, other=0.0).to(tl.float32)
    v = v + b2[None, :]

    v_3d = tl.reshape(v, (BLOCK_ROWS, BLOCK_GROUPS, CHANNELS_PER_GROUP))
    mean = tl.sum(v_3d, axis=2) / float(CHANNELS_PER_GROUP)
    centered = v_3d - mean[:, :, None]
    var = tl.sum(centered * centered, axis=2) / float(CHANNELS_PER_GROUP)
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + c_idx, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c_idx, mask=col_mask, other=0.0).to(tl.float32)
    gamma_3d = tl.reshape(gamma, (BLOCK_GROUPS, CHANNELS_PER_GROUP))
    beta_3d = tl.reshape(beta, (BLOCK_GROUPS, CHANNELS_PER_GROUP))

    out = centered * rstd[:, :, None]
    out = out * gamma_3d[None, :, :]
    out = out + beta_3d[None, :, :]

    out_2d = tl.reshape(out, (BLOCK_ROWS, BLOCK_GROUPS * CHANNELS_PER_GROUP))
    tl.store(y_ptrs, out_2d.to(y_ptr.dtype.element_ty), mask=mask)


def _post_gemm_fused(
    x_linear: torch.Tensor,
    b_add: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x_linear.ndim == 2
    n_rows, c = x_linear.shape
    assert c % num_groups == 0
    channels_per_group = c // num_groups

    assert x_linear.device.type == "xpu"
    for t in (b_add, gn_weight, gn_bias):
        assert t.device == x_linear.device
        assert t.dtype == x_linear.dtype

    y = torch.empty_like(x_linear)
    stride_xn, stride_xc = x_linear.stride()
    stride_yn, stride_yc = y.stride()

    def grid(meta):
        return (
            triton.cdiv(n_rows, meta["BLOCK_ROWS"]),
            triton.cdiv(num_groups, meta["BLOCK_GROUPS"]),
        )

    _swish_bias_groupnorm_kernel[grid](
        x_linear, b_add, gn_weight, gn_bias, y,
        n_rows, c,
        stride_xn, stride_xc,
        stride_yn, stride_yc,
        eps,
        num_groups,
        CHANNELS_PER_GROUP=channels_per_group,
        LOG2E=1.4426950408889634,
        grf_mode="auto",
    )
    return y


# -----------------------------------------------------------------------------
# Top-level kernel_function
# -----------------------------------------------------------------------------
def kernel_function(
    x: torch.Tensor,
    W: torch.Tensor,
    b_linear: torch.Tensor,
    b_add: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
    W_t: torch.Tensor = None,
) -> torch.Tensor:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU backend is required.")

    target_dtype = W.dtype

    x_xpu = x
    if x_xpu.device.type != "xpu" or x_xpu.dtype != target_dtype:
        x_xpu = x_xpu.to("xpu", dtype=target_dtype)
    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()

    W_xpu = W
    if W_xpu.device.type != "xpu" or W_xpu.dtype != target_dtype:
        W_xpu = W_xpu.to("xpu", dtype=target_dtype)
    if not W_xpu.is_contiguous():
        W_xpu = W_xpu.contiguous()

    b_linear_xpu = b_linear
    if b_linear_xpu.device.type != "xpu" or b_linear_xpu.dtype != target_dtype:
        b_linear_xpu = b_linear_xpu.to("xpu", dtype=target_dtype)
    if not b_linear_xpu.is_contiguous():
        b_linear_xpu = b_linear_xpu.contiguous()

    b_add_xpu = b_add
    if b_add_xpu.device.type != "xpu" or b_add_xpu.dtype != target_dtype:
        b_add_xpu = b_add_xpu.to("xpu", dtype=target_dtype)
    if not b_add_xpu.is_contiguous():
        b_add_xpu = b_add_xpu.contiguous()

    gn_weight_xpu = gn_weight
    if gn_weight_xpu.device.type != "xpu" or gn_weight_xpu.dtype != target_dtype:
        gn_weight_xpu = gn_weight_xpu.to("xpu", dtype=target_dtype)
    if not gn_weight_xpu.is_contiguous():
        gn_weight_xpu = gn_weight_xpu.contiguous()

    gn_bias_xpu = gn_bias
    if gn_bias_xpu.device.type != "xpu" or gn_bias_xpu.dtype != target_dtype:
        gn_bias_xpu = gn_bias_xpu.to("xpu", dtype=target_dtype)
    if not gn_bias_xpu.is_contiguous():
        gn_bias_xpu = gn_bias_xpu.contiguous()

    mid = torch.nn.functional.linear(x_xpu, W_xpu, b_linear_xpu)
    return _post_gemm_fused(mid, b_add_xpu, gn_weight_xpu, gn_bias_xpu, num_groups, eps=1e-5)


batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups

        self._cached_params_ready = False
        self._cached_weight_version = -1
        self._cached_bias_version = -1
        self._cached_gn_weight_version = -1
        self._cached_gn_bias_version = -1
        self._packed_weight_t = None

    def _ensure_xpu_params(self):
        target_dtype = self.matmul.weight.dtype

        if self.matmul.weight.device.type != "xpu":
            self.matmul.weight.data = self.matmul.weight.data.to("xpu", dtype=target_dtype).contiguous()
        elif self.matmul.weight.dtype != target_dtype or not self.matmul.weight.is_contiguous():
            self.matmul.weight.data = self.matmul.weight.data.to(dtype=target_dtype).contiguous()

        if self.matmul.bias is not None:
            if self.matmul.bias.device.type != "xpu":
                self.matmul.bias.data = self.matmul.bias.data.to("xpu", dtype=target_dtype).contiguous()
            elif self.matmul.bias.dtype != target_dtype or not self.matmul.bias.is_contiguous():
                self.matmul.bias.data = self.matmul.bias.data.to(dtype=target_dtype).contiguous()

        if self.bias.device.type != "xpu":
            self.bias.data = self.bias.data.to("xpu", dtype=target_dtype).contiguous()
        elif self.bias.dtype != target_dtype or not self.bias.is_contiguous():
            self.bias.data = self.bias.data.to(dtype=target_dtype).contiguous()

        if self.group_norm.weight.device.type != "xpu":
            self.group_norm.weight.data = self.group_norm.weight.data.to("xpu", dtype=target_dtype).contiguous()
        elif self.group_norm.weight.dtype != target_dtype or not self.group_norm.weight.is_contiguous():
            self.group_norm.weight.data = self.group_norm.weight.data.to(dtype=target_dtype).contiguous()

        if self.group_norm.bias.device.type != "xpu":
            self.group_norm.bias.data = self.group_norm.bias.data.to("xpu", dtype=target_dtype).contiguous()
        elif self.group_norm.bias.dtype != target_dtype or not self.group_norm.bias.is_contiguous():
            self.group_norm.bias.data = self.group_norm.bias.data.to(dtype=target_dtype).contiguous()

        self._packed_weight_t = self.matmul.weight.t().contiguous()

        self._cached_params_ready = True
        self._cached_weight_version = int(self.matmul.weight._version)
        self._cached_bias_version = int(self.bias._version)
        self._cached_gn_weight_version = int(self.group_norm.weight._version)
        self._cached_gn_bias_version = int(self.group_norm.bias._version)

    def _params_need_refresh(self):
        if not self._cached_params_ready:
            return True
        if self.matmul.weight.device.type != "xpu":
            return True
        if self.bias.device.type != "xpu":
            return True
        if self.group_norm.weight.device.type != "xpu":
            return True
        if self.group_norm.bias.device.type != "xpu":
            return True
        if int(self.matmul.weight._version) != self._cached_weight_version:
            return True
        if int(self.bias._version) != self._cached_bias_version:
            return True
        if int(self.group_norm.weight._version) != self._cached_gn_weight_version:
            return True
        if int(self.group_norm.bias._version) != self._cached_gn_bias_version:
            return True
        if self._packed_weight_t is None:
            return True
        return False

    def forward(self, x):
        if self._params_need_refresh():
            self._ensure_xpu_params()

        return kernel_function(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self._packed_weight_t,
        )