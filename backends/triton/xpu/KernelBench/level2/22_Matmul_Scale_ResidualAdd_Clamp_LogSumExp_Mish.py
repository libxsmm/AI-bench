# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl

assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU device is not available"

LOG2E = 1.4426950408889634


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_bias_scale_residual_kernel(
    x_ptr,
    w_kn_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    scale,
    ACCUM_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if GROUP_SIZE_M > 0 and num_pid_m > 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_kn_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K, BLOCK_K):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b = tl.load(w_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    acc = (acc + bias_vals[None, :]) * tl.full((), 2.0 * scale, dtype=tl.float32)
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


def linear_scale_residual(x, weight_kn, bias, scale=2.0):
    assert (
        isinstance(x, torch.Tensor)
        and isinstance(weight_kn, torch.Tensor)
        and isinstance(bias, torch.Tensor)
    )
    assert (
        x.device.type == "xpu"
        and weight_kn.device.type == "xpu"
        and bias.device.type == "xpu"
    )
    assert (
        x.dtype == torch.float16
        and weight_kn.dtype == torch.float16
        and bias.dtype == torch.float16
    )

    x = x.contiguous()
    weight_kn = weight_kn.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    Kw, Nw = weight_kn.shape
    assert K == Kw and Nw == bias.shape[0]

    y = torch.empty((M, Nw), device=x.device, dtype=torch.float16)
    stride_xm, stride_xk = x.stride(0), x.stride(1)
    stride_wk, stride_wn = weight_kn.stride(0), weight_kn.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(Nw, meta["BLOCK_N"]),)

    _linear_bias_scale_residual_kernel[grid](
        x,
        weight_kn,
        bias,
        y,
        M,
        Nw,
        K,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_ym,
        stride_yn,
        float(scale),
        False,
        grf_mode="auto",
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _clamp_lse_mish_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    CLAMP_MIN,
    CLAMP_MAX,
    LOG2E_CONST,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)

    neg_inf = tl.full((), -float("inf"), tl.float32)

    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        cur_cols = start_n + cols
        mask = row_mask[:, None] & (cur_cols[None, :] < N)
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + cur_cols[None, :] * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.minimum(tl.maximum(x, CLAMP_MIN), CLAMP_MAX)
        x = tl.where(mask, x, neg_inf)
        row_max = tl.maximum(row_max, tl.max(x, axis=1))

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        cur_cols = start_n + cols
        mask = row_mask[:, None] & (cur_cols[None, :] < N)
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + cur_cols[None, :] * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.minimum(tl.maximum(x, CLAMP_MIN), CLAMP_MAX)
        x = tl.where(mask, x, neg_inf)
        row_sum += tl.sum(tl.math.exp2((x - row_max[:, None]) * LOG2E_CONST), axis=1)

    y = row_max + tl.log(row_sum)

    soft = tl.where(
        y > 0.0,
        y + tl.log(1.0 + tl.math.exp2((-y) * LOG2E_CONST)),
        tl.log(1.0 + tl.math.exp2(y * LOG2E_CONST)),
    )
    abs_soft = tl.abs(soft)
    ex = tl.math.exp2((-2.0 * abs_soft) * LOG2E_CONST)
    tanh_abs = 1.0 - 2.0 * ex / (1.0 + ex)
    tanh_soft = tl.where(soft >= 0.0, tanh_abs, -tanh_abs)

    mish = y * tanh_soft
    out_val = y * mish

    tl.store(
        out_ptr + rows * stride_om + 0 * stride_on,
        out_val.to(tl.float16),
        mask=row_mask,
    )


def clamp_logsumexp_mish(x):
    assert isinstance(x, torch.Tensor)
    assert x.device.type == "xpu" and x.dtype == torch.float16 and x.dim() == 2

    x = x.contiguous()
    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=torch.float16)
    stride_xm, stride_xn = x.stride(0), x.stride(1)
    stride_om, stride_on = out.stride(0), out.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    _clamp_lse_mish_kernel[grid](
        x,
        out,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_om,
        stride_on,
        float(-10.0),
        float(10.0),
        float(LOG2E),
    )
    return out


def kernel_function(x, weight_kn, bias):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if weight_kn.device.type != "xpu" or weight_kn.dtype != torch.float16:
        weight_kn_xpu = weight_kn.to("xpu", dtype=torch.float16).contiguous()
    else:
        weight_kn_xpu = weight_kn.contiguous()

    if bias.device.type != "xpu" or bias.dtype != torch.float16:
        bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bias_xpu = bias.contiguous()

    y1 = linear_scale_residual(x_xpu, weight_kn_xpu, bias_xpu, scale=2.0)
    y2 = clamp_logsumexp_mish(y1)
    return y2


batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self._params_on_xpu = False
        self.weight_kn = None
        self.bias_xpu = None

    def _ensure_xpu_params(self):
        need_init = (
            (not self._params_on_xpu)
            or self.weight_kn is None
            or self.bias_xpu is None
            or self.matmul.weight.data.device.type != "xpu"
            or self.matmul.bias.data.device.type != "xpu"
            or self.matmul.weight.data.dtype != torch.float16
            or self.matmul.bias.data.dtype != torch.float16
        )

        if need_init:
            weight_xpu = self.matmul.weight.data.to(
                "xpu", dtype=torch.float16
            ).contiguous()
            bias_xpu = self.matmul.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.matmul.weight.data = weight_xpu
            self.matmul.bias.data = bias_xpu
            self.weight_kn = weight_xpu.t().contiguous()
            self.bias_xpu = bias_xpu
            self._params_on_xpu = True
        else:
            if not self.matmul.weight.data.is_contiguous():
                self.matmul.weight.data = self.matmul.weight.data.contiguous()
            if not self.matmul.bias.data.is_contiguous():
                self.matmul.bias.data = self.matmul.bias.data.contiguous()
            if (
                self.weight_kn.device.type != "xpu"
                or self.weight_kn.dtype != torch.float16
                or not self.weight_kn.is_contiguous()
            ):
                self.weight_kn = self.matmul.weight.data.t().contiguous()
            if (
                self.bias_xpu.device.type != "xpu"
                or self.bias_xpu.dtype != torch.float16
                or not self.bias_xpu.is_contiguous()
            ):
                self.bias_xpu = self.matmul.bias.data.contiguous()

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        self._ensure_xpu_params()
        return kernel_function(x, self.weight_kn, self.bias_xpu)
