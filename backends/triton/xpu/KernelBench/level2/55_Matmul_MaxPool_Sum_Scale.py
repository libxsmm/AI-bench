# ruff: noqa: E731
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _fused_linear_maxpool1d_configs():
    # Keep original safe configs and add broader XPU-oriented exploration.
    # Avoid grf_mode inside triton.Config() per XPU backend constraint.
    return [
        # original / conservative
        triton.Config({"BM": 32, "BNP": 32, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BNP": 32, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=3),
        triton.Config({"BM": 32, "BNP": 64, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BNP": 64, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=3),
        triton.Config({"BM": 32, "BNP": 32, "BK": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=3),
        # suggested / medium
        triton.Config({"BM": 64, "BNP": 64, "BK": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BM": 64, "BNP": 128, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BM": 64, "BNP": 128, "BK": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BNP": 64, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BNP": 64, "BK": 64, "GROUP_SIZE_M": 1}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BNP": 128, "BK": 16, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=3),
        triton.Config({"BM": 128, "BNP": 128, "BK": 32, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        triton.Config({"BM": 128, "BNP": 128, "BK": 64, "GROUP_SIZE_M": 1}, num_warps=16, num_stages=2),
        # swizzle alternatives
        triton.Config({"BM": 64, "BNP": 128, "BK": 32, "GROUP_SIZE_M": 2}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BNP": 128, "BK": 32, "GROUP_SIZE_M": 2}, num_warps=16, num_stages=2),
        triton.Config({"BM": 128, "BNP": 128, "BK": 32, "GROUP_SIZE_M": 4}, num_warps=16, num_stages=2),
        # required large-tile XPU exploration, including 256x256 / 32-warps
        triton.Config({"BM": 256, "BNP": 128, "BK": 16, "GROUP_SIZE_M": 1}, num_warps=32, num_stages=2),
        triton.Config({"BM": 128, "BNP": 256, "BK": 16, "GROUP_SIZE_M": 1}, num_warps=32, num_stages=2),
        triton.Config({"BM": 256, "BNP": 256, "BK": 16, "GROUP_SIZE_M": 1}, num_warps=32, num_stages=2),
    ]


def _row_sum_scale_configs():
    # Reduction-specific search space.
    return [
        triton.Config({"BLOCK_M": 8, "BLOCK_SIZE_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_SIZE_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_SIZE_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_SIZE_C": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_SIZE_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_SIZE_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_SIZE_C": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_SIZE_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_SIZE_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_SIZE_C": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_SIZE_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_SIZE_C": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_SIZE_C": 512}, num_warps=16, num_stages=3),
    ]


class Model(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self._xpu_ready = False

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        if not self._xpu_ready:
            if self.matmul.weight.device.type != "xpu" or self.matmul.weight.dtype != torch.float16:
                self.matmul.weight.data = self.matmul.weight.data.to("xpu", dtype=torch.float16).contiguous()
            else:
                self.matmul.weight.data = self.matmul.weight.data.contiguous()

            if self.matmul.bias is not None:
                if self.matmul.bias.device.type != "xpu" or self.matmul.bias.dtype != torch.float16:
                    self.matmul.bias.data = self.matmul.bias.data.to("xpu", dtype=torch.float16).contiguous()
                else:
                    self.matmul.bias.data = self.matmul.bias.data.contiguous()
            self._xpu_ready = True

        return kernel_function(x, self.matmul.weight, self.matmul.bias, self.scale_factor)


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]


@triton.autotune(
    configs=_fused_linear_maxpool1d_configs(),
    key=["M", "N_OUT", "K"],
)
@triton.jit
def _fused_linear_maxpool1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    M,
    N_OUT,
    K,
    stride_xm,
    stride_xk,
    stride_wo,
    stride_wk,
    stride_om,
    stride_on,
    BM: tl.constexpr,
    BNP: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_pool = N_OUT // 2

    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(n_pool, BNP)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_m)
    pid_np = (pid % num_pid_in_group) // group_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_np = pid_np * BNP + tl.arange(0, BNP)

    j0 = offs_np * 2
    j1 = j0 + 1

    mask_m = offs_m < M
    mask_np = offs_np < n_pool
    mask_j0 = j0 < N_OUT
    mask_j1 = j1 < N_OUT

    acc0 = tl.zeros((BM, BNP), dtype=tl.float32)
    acc1 = tl.zeros((BM, BNP), dtype=tl.float32)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    w0_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(N_OUT, K),
        strides=(stride_wo, stride_wk),
        offsets=(pid_np * BNP * 2, 0),
        block_shape=(BNP, BK),
        order=(1, 0),
    )
    w1_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(N_OUT, K),
        strides=(stride_wo, stride_wk),
        offsets=(pid_np * BNP * 2 + 1, 0),
        block_shape=(BNP, BK),
        order=(1, 0),
    )

    k_tiles = tl.cdiv(K, BK)
    for _ in range(0, k_tiles):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b0 = tl.load(w0_bp, boundary_check=(0, 1))
        b1 = tl.load(w1_bp, boundary_check=(0, 1))

        acc0 += tl.dot(a, tl.trans(b0))
        acc1 += tl.dot(a, tl.trans(b1))

        x_bp = tl.advance(x_bp, (0, BK))
        w0_bp = tl.advance(w0_bp, (0, BK))
        w1_bp = tl.advance(w1_bp, (0, BK))

    bias0 = tl.load(b_ptr + j0, mask=mask_j0, other=0.0).to(tl.float32)
    bias1 = tl.load(b_ptr + j1, mask=mask_j1, other=0.0).to(tl.float32)

    acc0 = acc0 + bias0[None, :]
    acc1 = acc1 + bias1[None, :]
    pooled = tl.maximum(acc0, acc1)

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_np[None, :] * stride_on
    tl.store(o_ptrs, pooled.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_np[None, :])


@triton.autotune(
    configs=_row_sum_scale_configs(),
    key=["N", "C"],
)
@triton.jit
def _row_sum_scale_kernel(
    x_ptr,
    y_ptr,
    N, C,
    stride_xn, stride_xc,
    stride_yn,
    scale: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_ctiles = tl.cdiv(C, BLOCK_SIZE_C)

    for ct in range(0, num_ctiles):
        offs_c = ct * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xn + offs_c[None, :] * stride_xc
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_c[None, :] < C), other=0.0)
        acc += tl.sum(x_tile.to(tl.float32), axis=1)

    out = (acc * scale).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m * stride_yn, out, mask=mask_m)


def fused_linear_maxpool1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "xpu"
    assert weight.device == x.device and bias.device == x.device
    assert x.dtype == torch.float16 and weight.dtype == torch.float16 and bias.dtype == torch.float16
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1

    M, K = x.shape
    N_OUT, K_w = weight.shape
    assert K == K_w
    assert bias.shape[0] == N_OUT

    N_POOL = N_OUT // 2
    out = torch.empty((M, N_POOL), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N_POOL, meta["BNP"]),)

    _fused_linear_maxpool1d_kernel[grid](
        x, weight, bias, out,
        M, N_OUT, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def row_sum_scale(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    assert x.device.type == "xpu"
    N, C = x.shape
    y = torch.empty((N,), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]),)

    _row_sum_scale_kernel[grid](
        x, y,
        N, C,
        x.stride(0), x.stride(1),
        y.stride(0),
        float(scale_factor),
    )
    return y


def kernel_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if weight.device.type != "xpu" or weight.dtype != torch.float16:
        w_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
    else:
        w_xpu = weight.contiguous()

    if bias.device.type != "xpu" or bias.dtype != torch.float16:
        b_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        b_xpu = bias.contiguous()

    mid = fused_linear_maxpool1d(x_xpu, w_xpu, b_xpu)
    out = row_sum_scale(mid, float(scale_factor))
    return out


def run_test():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU device is not available. Skipping test.")
        sys.exit(0)

    in_f, out_f, ksz, scale = get_init_inputs()
    x = get_inputs()[0].to("xpu", dtype=torch.float16)

    model = Model(in_f, out_f, ksz, scale)
    y_triton = model(x)

    w = model.matmul.weight.to("xpu", dtype=torch.float16)
    b = model.matmul.bias.to("xpu", dtype=torch.float16)
    y_ref = torch.nn.functional.linear(x, w, b)
    y_ref = torch.maximum(y_ref[:, 0::2], y_ref[:, 1::2])
    y_ref = torch.sum(y_ref, dim=1)
    y_ref = y_ref * scale

    ok = torch.allclose(y_triton, y_ref, rtol=1e-2, atol=1e-2)
    if not ok:
        max_diff = torch.max(torch.abs(y_triton - y_ref)).detach().cpu()
        print(f"Test FAILED: max difference {max_diff}")
        sys.exit(1)
    print("PASS")
