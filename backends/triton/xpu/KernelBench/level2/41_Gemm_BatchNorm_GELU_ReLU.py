# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _linear_gelu_relu_autotune_configs():
    configs = []

    # XPU-focused search space:
    # - mandatory 256x256 / 32-warps coverage
    # - asymmetric 256x128 / 128x256 fallbacks
    # - medium and small tiles for shape variation
    shape_specs = [
        ((256, 256, 16), [(32, 2), (32, 3), (32, 4), (16, 3)], (1, 4)),
        ((256, 256, 32), [(32, 2), (32, 3), (32, 4), (16, 3)], (1, 4)),
        ((256, 128, 16), [(16, 2), (16, 3), (32, 3)], (1, 4, 8)),
        ((256, 128, 32), [(16, 2), (16, 3), (32, 3)], (1, 4, 8)),
        ((128, 256, 16), [(16, 2), (16, 3), (32, 3)], (1, 4, 8)),
        ((128, 256, 32), [(16, 2), (16, 3), (32, 3)], (1, 4, 8)),
        ((128, 128, 32), [(8, 2), (16, 2), (16, 3)], (1, 4, 8)),
        ((128, 128, 64), [(8, 2), (16, 2), (16, 3)], (1, 4, 8)),
        ((64, 256, 32), [(8, 2), (16, 2), (16, 3)], (1, 4, 8)),
        ((256, 64, 32), [(8, 2), (16, 2), (16, 3)], (1, 4, 8)),
        ((64, 128, 32), [(8, 2), (8, 3), (16, 2)], (1, 8)),
        ((128, 64, 32), [(8, 2), (8, 3), (16, 2)], (1, 8)),
        ((64, 64, 32), [(4, 2), (8, 2), (8, 3)], (1, 8)),
        ((64, 64, 64), [(4, 2), (8, 2), (8, 3)], (1, 8)),
    ]

    for (bm, bn, bk), warp_stage_pairs, group_sizes in shape_specs:
        for gs in group_sizes:
            for nw, ns in warp_stage_pairs:
                configs.append(
                    triton.Config(
                        {
                            "GROUP_SIZE_M": gs,
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                        },
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


@triton.jit
def _erf_approx(x):
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    sign = tl.where(x >= 0, 1.0, -1.0)
    ax = tl.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    y = 1.0 - y * tl.exp(-ax * ax)
    return sign * y


@triton.jit
def _linear_bn_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

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
        shape=(N, K),
        strides=(stride_wn, stride_wk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b = tl.load(w_bp, boundary_check=(0, 1))
        acc += tl.dot(a, tl.trans(b))
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (0, BLOCK_K))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    b_vec = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    mean_vec = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    var_vec = tl.load(var_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    gamma_vec = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    beta_vec = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_vec + eps)
    scale = gamma_vec * inv_std
    shift = beta_vec + (b_vec - mean_vec) * scale
    y_out = acc * scale[None, :] + shift[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, y_out.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _gelu_relu_kernel(
    x_ptr,
    y_ptr,
    N,
    M,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < N) & (offs_n[None, :] < M)
    x_off = x_ptr + offs_m[:, None] * stride_x0 + offs_n[None, :] * stride_x1
    y_off = y_ptr + offs_m[:, None] * stride_y0 + offs_n[None, :] * stride_y1
    x = tl.load(x_off, mask=mask, other=0.0).to(tl.float32)
    t = x * 0.7071067811865476
    erf_t = _erf_approx(t)
    gelu = 0.5 * x * (1.0 + erf_t)
    y_val = tl.maximum(gelu, 0.0).to(y_ptr.dtype.element_ty)
    tl.store(y_off, y_val, mask=mask)


@triton.autotune(
    configs=_linear_gelu_relu_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _linear_gelu_relu_kernel(
    x_ptr,
    w_ptr,
    shift_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if GROUP_SIZE_M > 0:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

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
        shape=(N, K),
        strides=(stride_wn, stride_wk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b = tl.load(w_bp, boundary_check=(0, 1))
        acc += tl.dot(a, tl.trans(b))
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (0, BLOCK_K))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + shift[None, :]

    t = acc * 0.7071067811865476
    erf_t = _erf_approx(t)
    gelu = 0.5 * acc * (1.0 + erf_t)
    out = tl.maximum(gelu, 0.0)

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, out.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


def _prepare_xpu_fp16(t):
    if t.device.type != "xpu" or t.dtype != torch.float16:
        t = t.to("xpu", dtype=torch.float16)
    return t.contiguous()


def _prepare_xpu_fp32(t):
    if t.device.type != "xpu" or t.dtype != torch.float32:
        t = t.to("xpu", dtype=torch.float32)
    return t.contiguous()


def _linear_bn(x, w, b, gamma, beta, mean, var, eps):
    x_xpu = _prepare_xpu_fp16(x)
    w_xpu = _prepare_xpu_fp16(w)
    b_xpu = _prepare_xpu_fp16(b)
    gamma_xpu = _prepare_xpu_fp16(gamma)
    beta_xpu = _prepare_xpu_fp16(beta)
    mean_xpu = _prepare_xpu_fp16(mean)
    var_xpu = _prepare_xpu_fp16(var)

    M, K = x_xpu.shape
    N, K_w = w_xpu.shape
    assert K == K_w

    y = torch.empty((M, N), device=x_xpu.device, dtype=torch.float16)
    stride_xm, stride_xk = x_xpu.stride(0), x_xpu.stride(1)
    stride_wn, stride_wk = w_xpu.stride(0), w_xpu.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_bn_fwd_kernel[grid](
        x_xpu,
        w_xpu,
        b_xpu,
        gamma_xpu,
        beta_xpu,
        mean_xpu,
        var_xpu,
        y,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        float(eps),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=32,
        num_stages=3,
    )
    return y


def _gelu_relu(x):
    x_xpu = _prepare_xpu_fp16(x)
    N, M = x_xpu.shape
    y = torch.empty_like(x_xpu)
    s0, s1 = x_xpu.stride(0), x_xpu.stride(1)
    s0y, s1y = y.stride(0), y.stride(1)
    BLOCK_M, BLOCK_N = 128, 128
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    _gelu_relu_kernel[grid](
        x_xpu,
        y,
        N,
        M,
        s0,
        s1,
        s0y,
        s1y,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return y


def kernel_function(x, w_fold, shift):
    x_xpu = _prepare_xpu_fp16(x)
    w_fold_xpu = _prepare_xpu_fp16(w_fold)
    shift_xpu = _prepare_xpu_fp32(shift)

    M, K = x_xpu.shape
    N, K_w = w_fold_xpu.shape
    assert K == K_w
    assert shift_xpu.shape == (N,)

    y = torch.empty((M, N), device=x_xpu.device, dtype=torch.float16)
    stride_xm, stride_xk = x_xpu.stride(0), x_xpu.stride(1)
    stride_wn, stride_wk = w_fold_xpu.stride(0), w_fold_xpu.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _linear_gelu_relu_kernel[grid](
        x_xpu,
        w_fold_xpu,
        shift_xpu,
        y,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        grf_mode="auto",
    )
    return y


batch_size = 16384
in_features = 4096
out_features = 4096


def get_init_inputs():
    return [in_features, out_features]


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self._cached_w_fold = None
        self._cached_shift = None
        self._cache_key = None

    def _ensure_folded(self):
        w = self.linear.weight
        b = self.linear.bias
        gamma = self.bn.weight
        beta = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var

        key = (
            int(w._version),
            int(b._version),
            int(gamma._version),
            int(beta._version),
            int(mean._version),
            int(var._version),
            w.device.type,
            b.device.type,
            gamma.device.type,
            beta.device.type,
            mean.device.type,
            var.device.type,
        )
        if (
            self._cache_key == key
            and self._cached_w_fold is not None
            and self._cached_shift is not None
        ):
            return

        w_fp32 = w.detach().to("xpu", dtype=torch.float32).contiguous()
        b_fp32 = b.detach().to("xpu", dtype=torch.float32).contiguous()
        gamma_fp32 = gamma.detach().to("xpu", dtype=torch.float32).contiguous()
        beta_fp32 = beta.detach().to("xpu", dtype=torch.float32).contiguous()
        mean_fp32 = mean.detach().to("xpu", dtype=torch.float32).contiguous()
        var_fp32 = var.detach().to("xpu", dtype=torch.float32).contiguous()

        scale = gamma_fp32 / torch.sqrt(var_fp32 + 1e-5)
        self._cached_w_fold = (w_fp32 * scale[:, None]).to(torch.float16).contiguous()
        self._cached_shift = (beta_fp32 + (b_fp32 - mean_fp32) * scale).contiguous()
        self._cache_key = key

    def forward(self, x):
        self._ensure_folded()
        return kernel_function(x, self._cached_w_fold, self._cached_shift)
