# ruff: noqa: E731
import math
import torch
import triton
import triton.language as tl
import torch.nn as nn


def _get_configs():
    return [
        # Large XPU-oriented configs for the main compute-bound regime
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=3,
        ),
        # Small / fallback configs for shape changes
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
            num_warps=8,
            num_stages=2,
        ),
    ]


@triton.autotune(
    configs=_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _linear_relu_div_kernel(
    x_ptr,
    w_ptr,
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
    inv_divisor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

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
        a = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc=acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    acc = tl.maximum(acc, 0.0) * inv_divisor

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


@triton.jit
def _relu_mul_kernel(
    y_ptr,
    b_ptr,
    M,
    N,
    stride_ym,
    stride_yn,
    inv_divisor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = tl.maximum(acc + bias[None, :], 0.0) * inv_divisor
    tl.store(ptrs, acc.to(tl.float16), mask=mask)


def _normalize_divisor(divisor):
    if isinstance(divisor, (int, float)):
        divisor_val = float(divisor)
    else:
        raise TypeError("divisor must be a Python int or float to avoid device-host sync")
    if divisor_val == 0.0:
        raise ValueError("divisor must be non-zero")
    return divisor_val


def kernel_function(x, weight, bias, divisor, weight_is_packed_kn=False):
    assert isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)

    divisor_val = _normalize_divisor(divisor)

    x_xpu = x.to("xpu", dtype=torch.float16).contiguous() if (
        x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous()
    ) else x
    b_xpu = bias.to("xpu", dtype=torch.float16).contiguous() if (
        bias.device.type != "xpu" or bias.dtype != torch.float16 or not bias.is_contiguous()
    ) else bias

    if weight_is_packed_kn:
        w_t_xpu = weight.to("xpu", dtype=torch.float16).contiguous() if (
            weight.device.type != "xpu" or weight.dtype != torch.float16 or not weight.is_contiguous()
        ) else weight
    else:
        if weight.device.type == "xpu" and weight.dtype == torch.float16 and weight.is_contiguous():
            w_t_xpu = weight.t().contiguous()
        else:
            w_t_xpu = weight.to("xpu", dtype=torch.float16).t().contiguous()

    M_dim, K_dim = x_xpu.shape
    K_w, N_dim = w_t_xpu.shape

    assert w_t_xpu.ndim == 2 and b_xpu.ndim == 1
    assert K_w == K_dim
    assert b_xpu.shape[0] == N_dim

    y = torch.empty((M_dim, N_dim), device=x_xpu.device, dtype=torch.float16)
    inv_divisor = 1.0 / divisor_val

    def grid_gemm(meta):
        return (triton.cdiv(M_dim, meta["BLOCK_M"]) * triton.cdiv(N_dim, meta["BLOCK_N"]),)

    _linear_relu_div_kernel[grid_gemm](
        x_xpu,
        w_t_xpu,
        b_xpu,
        y,
        M_dim,
        N_dim,
        K_dim,
        x_xpu.stride(0),
        x_xpu.stride(1),
        w_t_xpu.stride(0),
        w_t_xpu.stride(1),
        y.stride(0),
        y.stride(1),
        inv_divisor,
    )
    return y


batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, divisor]


class Model(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.divisor = _normalize_divisor(divisor)
        self._xpu_params_ready = False
        self.weight_kn = None

    def _ensure_xpu_params(self):
        if not self._xpu_params_ready:
            weight_xpu = self.gemm.weight.data.to("xpu", dtype=torch.float16).contiguous()
            bias_xpu = self.gemm.bias.data.to("xpu", dtype=torch.float16).contiguous()

            self.gemm.weight.data = weight_xpu
            self.gemm.bias.data = bias_xpu
            self.weight_kn = weight_xpu.t().contiguous()
            self._xpu_params_ready = True

    def forward(self, x):
        self._ensure_xpu_params()
        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to("xpu", dtype=torch.float16).contiguous()

        return kernel_function(
            x,
            self.weight_kn,
            self.gemm.bias,
            self.divisor,
            weight_is_packed_kn=True,
        )