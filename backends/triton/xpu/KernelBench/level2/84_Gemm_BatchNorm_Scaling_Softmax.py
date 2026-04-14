# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_bn_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    gamma_ptr,
    beta_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_m)
    pid_n = (pid % num_pid_in_group) // group_m

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

    for _ in range(0, K, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b = tl.load(w_bp, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc=acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    lin_bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    mean = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    rstd = tl.rsqrt(var + EPS)
    acc = acc + lin_bias[None, :]
    acc = (acc - mean[None, :]) * rstd[None, :]
    acc = acc * gamma[None, :] + beta[None, :]

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_bp, acc.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def _scale_softmax_rowwise_kernel_contig(
    x_ptr,
    y_ptr,
    scale_ptr,
    N,
    C,
    stride_xn,
    stride_xc,
    stride_yn,
    stride_yc,
    BLOCK_SIZE: tl.constexpr,
    LOG2E: tl.constexpr = 1.4426950408889634,
):
    row = tl.program_id(0)
    if row >= N:
        return

    row64 = row.to(tl.int64)
    x_row = x_ptr + row64 * stride_xn
    y_row = y_ptr + row64 * stride_yn
    scale = tl.load(scale_ptr).to(tl.float32)
    cols = tl.arange(0, BLOCK_SIZE)
    cols = tl.max_contiguous(cols, BLOCK_SIZE)

    max_val = tl.full((), -float("inf"), tl.float32)
    for start in tl.range(0, C, BLOCK_SIZE):
        offs = start + cols
        mask = offs < C
        vals = tl.load(x_row + offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
        logits = vals * scale
        logits = tl.where(mask, logits, -float("inf"))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))

    sum_val = tl.zeros((), tl.float32)
    for start in tl.range(0, C, BLOCK_SIZE):
        offs = start + cols
        mask = offs < C
        vals = tl.load(x_row + offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
        logits = vals * scale - max_val
        logits = tl.where(mask, logits, -float("inf"))
        exp_logits = tl.math.exp2(logits * LOG2E)
        exp_logits = tl.where(mask, exp_logits, 0.0)
        sum_val += tl.sum(exp_logits, axis=0)

    inv_sum = 1.0 / sum_val
    for start in tl.range(0, C, BLOCK_SIZE):
        offs = start + cols
        mask = offs < C
        vals = tl.load(x_row + offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
        logits = vals * scale - max_val
        logits = tl.where(mask, logits, -float("inf"))
        exp_logits = tl.math.exp2(logits * LOG2E)
        out = tl.where(mask, exp_logits * inv_sum, 0.0)
        tl.store(y_row + offs * stride_yc, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def _scale_softmax_rowwise_singlepass_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    N,
    C,
    stride_xn,
    stride_xc,
    stride_yn,
    stride_yc,
    BLOCK_SIZE: tl.constexpr,
    LOG2E: tl.constexpr = 1.4426950408889634,
):
    row = tl.program_id(0)
    if row >= N:
        return

    row64 = row.to(tl.int64)
    x_row = x_ptr + row64 * stride_xn
    y_row = y_ptr + row64 * stride_yn
    scale = tl.load(scale_ptr).to(tl.float32)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    vals = tl.load(x_row + offs * stride_xc, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    logits = vals * scale
    logits = tl.where(mask, logits, -float("inf"))

    row_max = tl.max(logits, axis=0)
    exp_logits = tl.math.exp2((logits - row_max) * LOG2E)
    exp_logits = tl.where(mask, exp_logits, 0.0)
    row_sum = tl.sum(exp_logits, axis=0)
    out = exp_logits / row_sum

    tl.store(y_row + offs * stride_yc, out.to(y_ptr.dtype.element_ty), mask=mask)


def _ensure_xpu_contiguous(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.device.type != "xpu" or t.dtype != dtype or not t.is_contiguous():
        return t.to("xpu", dtype=dtype).contiguous()
    return t


def kernel_function(
    x: torch.Tensor,
    w_fold: torch.Tensor,
    b_fold: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    x_xpu = _ensure_xpu_contiguous(x, torch.float16)
    scale_xpu = _ensure_xpu_contiguous(scale, torch.float32)

    if (
        w_fold.device.type != "xpu"
        or w_fold.dtype != torch.float16
        or not w_fold.is_contiguous()
    ):
        w_fold = w_fold.to("xpu", dtype=torch.float16).contiguous()
    if (
        b_fold.device.type != "xpu"
        or b_fold.dtype != torch.float16
        or not b_fold.is_contiguous()
    ):
        b_fold = b_fold.to("xpu", dtype=torch.float16).contiguous()

    out1 = F.linear(x_xpu, w_fold, b_fold)
    y = torch.empty_like(out1)

    n_rows, n_cols = out1.shape
    if n_cols <= 1024 and out1.stride(1) == 1 and y.stride(1) == 1:
        grid = (n_rows,)
        _scale_softmax_rowwise_singlepass_kernel[grid](
            out1,
            y,
            scale_xpu,
            n_rows,
            n_cols,
            out1.stride(0),
            out1.stride(1),
            y.stride(0),
            y.stride(1),
        )
    else:
        grid = (n_rows,)
        _scale_softmax_rowwise_kernel_contig[grid](
            out1,
            y,
            scale_xpu,
            n_rows,
            n_cols,
            out1.stride(0),
            out1.stride(1),
            y.stride(0),
            y.stride(1),
        )
    return y


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self._bn_eps = bn_eps

        self.register_buffer("_cached_w_fold", torch.empty(0))
        self.register_buffer("_cached_b_fold", torch.empty(0))

        self._cache_valid_py = False
        self._params_on_xpu = False
        self._fold_cache_version = None

    def _invalidate_fold_cache(self):
        self._cache_valid_py = False
        self._fold_cache_version = None

    def train(self, mode: bool = True):
        self._invalidate_fold_cache()
        return super().train(mode)

    def _ensure_params_on_xpu(self):
        if self._params_on_xpu:
            return
        self.gemm.weight.data = self.gemm.weight.data.to(
            "xpu", dtype=torch.float16
        ).contiguous()
        self.gemm.bias.data = self.gemm.bias.data.to(
            "xpu", dtype=torch.float16
        ).contiguous()
        self.bn.weight.data = self.bn.weight.data.to(
            "xpu", dtype=torch.float32
        ).contiguous()
        self.bn.bias.data = self.bn.bias.data.to(
            "xpu", dtype=torch.float32
        ).contiguous()
        self.bn.running_mean.data = self.bn.running_mean.data.to(
            "xpu", dtype=torch.float32
        ).contiguous()
        self.bn.running_var.data = self.bn.running_var.data.to(
            "xpu", dtype=torch.float32
        ).contiguous()
        self.scale.data = self.scale.data.to("xpu", dtype=torch.float32).contiguous()
        self._params_on_xpu = True

    def _current_fold_version(self):
        return (
            int(self.gemm.weight._version),
            int(self.gemm.bias._version),
            int(self.bn.weight._version),
            int(self.bn.bias._version),
            int(self.bn.running_mean._version),
            int(self.bn.running_var._version),
        )

    def _refresh_folded_params(self):
        self._ensure_params_on_xpu()

        w_xpu = self.gemm.weight
        b_xpu = self.gemm.bias
        gamma_xpu = self.bn.weight
        beta_xpu = self.bn.bias
        mean_xpu = self.bn.running_mean
        var_xpu = self.bn.running_var

        bn_scale_fp32 = gamma_xpu * torch.rsqrt(var_xpu + self._bn_eps)
        bn_scale_fp16 = bn_scale_fp32.to(torch.float16)

        self._cached_w_fold = (w_xpu * bn_scale_fp16[:, None]).contiguous()
        self._cached_b_fold = (
            (((b_xpu.to(torch.float32) - mean_xpu) * bn_scale_fp32) + beta_xpu)
            .to(torch.float16)
            .contiguous()
        )
        self._cache_valid_py = True
        self._fold_cache_version = self._current_fold_version()

    def forward(self, x):
        self._ensure_params_on_xpu()

        cur_ver = self._current_fold_version()
        if (
            (not self._cache_valid_py)
            or self._cached_w_fold.numel() == 0
            or self._cached_b_fold.numel() == 0
            or self._fold_cache_version != cur_ver
        ):
            self._refresh_folded_params()

        return kernel_function(
            x,
            self._cached_w_fold,
            self._cached_b_fold,
            self.scale,
        )
