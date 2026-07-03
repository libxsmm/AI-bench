import torch
import torch.nn as nn
import triton
import triton.language as tl


def _gemm_autotune_configs():
    configs = [
        # Original / conservative family
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        # Suggested XPU-oriented family
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 16}, num_warps=32, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=16, num_stages=2
        ),
        # Extra large-tile XPU candidates
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=32, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 16}, num_warps=32, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=32, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=32, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=3
        ),
    ]
    return configs


def _lse_autotune_configs():
    return [
        # Existing family
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=8, num_stages=2),
        # Expanded XPU search space
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=32, num_stages=2),
    ]


@triton.autotune(
    configs=_gemm_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _linear_bias_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(k_tiles):
        k_mask = offs_k < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        w_mask = k_mask[:, None] & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def _linear_bias_triton(x, weight, bias):
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw
    assert Nw == bias.shape[0]
    assert (
        x.device.type == "xpu"
        and weight.device.type == "xpu"
        and bias.device.type == "xpu"
    )
    assert x.dtype == weight.dtype
    assert x.dtype in (torch.bfloat16, torch.float16)

    y = torch.empty((M, Nw), device=x.device, dtype=x.dtype)
    stride_am, stride_ak = x.stride(0), x.stride(1)
    stride_wk, stride_wn = weight.stride(1), weight.stride(0)
    stride_cm, stride_cn = y.stride(0), y.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Nw, meta["BLOCK_N"]))

    _linear_bias_kernel[grid](
        x,
        weight,
        bias,
        y,
        M,
        Nw,
        Kx,
        stride_am,
        stride_ak,
        stride_wk,
        stride_wn,
        stride_cm,
        stride_cn,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=32, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _fused_lse_leaky_leaky_gelu_gelu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    in_bounds = pid < M
    offs_n = tl.arange(0, BLOCK_SIZE)

    base_x = x_ptr + pid * stride_xm
    base_o = out_ptr + pid * stride_om

    max_val = tl.full((), -float("inf"), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK_SIZE):
        idx = start + offs_n
        mask = in_bounds & (idx < N)
        x_block = tl.load(base_x + idx * stride_xn, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        blk_max = tl.max(x_block, axis=0)
        max_val = tl.maximum(max_val, blk_max)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK_SIZE):
        idx = start + offs_n
        mask = in_bounds & (idx < N)
        x_block = tl.load(base_x + idx * stride_xn, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        sum_exp += tl.sum(tl.exp(x_block - max_val), axis=0)

    lse = max_val + tl.log(sum_exp)
    lse = tl.where(lse >= 0, lse, negative_slope * lse)
    lse = tl.where(lse >= 0, lse, negative_slope * lse)

    inv_sqrt2 = 0.7071067811865476
    lse = 0.5 * lse * (1.0 + tl.math.erf(lse * inv_sqrt2))
    lse = 0.5 * lse * (1.0 + tl.math.erf(lse * inv_sqrt2))

    tl.store(base_o + 0 * stride_on, lse.to(out_ptr.dtype.element_ty), mask=in_bounds)


@triton.autotune(
    configs=_lse_autotune_configs(),
    key=["M", "N"],
)
@triton.jit
def _rowwise_lse_chain_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M

    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        cur_cols = start_n + cols
        mask = row_mask[:, None] & (cur_cols[None, :] < N)
        ptrs = x_ptr + rows[:, None] * stride_xm + cur_cols[None, :] * stride_xn
        x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=1))

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        cur_cols = start_n + cols
        mask = row_mask[:, None] & (cur_cols[None, :] < N)
        ptrs = x_ptr + rows[:, None] * stride_xm + cur_cols[None, :] * stride_xn
        x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max[:, None]), axis=1)

    lse = row_max + tl.log(row_sum)
    lse = tl.where(lse >= 0, lse, negative_slope * lse)
    lse = tl.where(lse >= 0, lse, negative_slope * lse)

    inv_sqrt2 = 0.7071067811865476
    lse = 0.5 * lse * (1.0 + tl.math.erf(lse * inv_sqrt2))
    lse = 0.5 * lse * (1.0 + tl.math.erf(lse * inv_sqrt2))

    out_ptrs = out_ptr + rows * stride_om
    tl.store(out_ptrs, lse.to(out_ptr.dtype.element_ty), mask=row_mask)


def _fused_lse_triton(x, negative_slope=0.01):
    assert x.ndim == 2
    assert x.device.type == "xpu"
    assert x.dtype in (torch.float16, torch.bfloat16)

    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    stride_xm, stride_xn = x.stride(0), x.stride(1)
    stride_om, stride_on = out.stride(0), out.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    _rowwise_lse_chain_kernel[grid](
        x,
        out,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_om,
        stride_on,
        negative_slope,
    )
    return out


def kernel_function(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available")

    if x.device.type == "xpu" and x.dtype == torch.float16 and x.is_contiguous():
        x_xpu = x
    else:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()

    if (
        weight.device.type == "xpu"
        and weight.dtype == x_xpu.dtype
        and weight.is_contiguous()
    ):
        weight_xpu = weight
    else:
        weight_xpu = weight.to("xpu", dtype=x_xpu.dtype).contiguous()

    bias_xpu = None
    if bias is not None:
        if (
            bias.device.type == "xpu"
            and bias.dtype == x_xpu.dtype
            and bias.is_contiguous()
        ):
            bias_xpu = bias
        else:
            bias_xpu = bias.to("xpu", dtype=x_xpu.dtype).contiguous()

    mid = torch.nn.functional.linear(x_xpu, weight_xpu, bias_xpu)
    return _fused_lse_triton(mid, negative_slope=0.01)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.bias = bias
        self._cached_weight_xpu = None
        self._cached_bias_xpu = None
        self._cached_weight_version = -1
        self._cached_bias_version = -1

    def _ensure_cached_params(self):
        w = self.gemm.weight
        w_ver = int(w._version)
        if (
            self._cached_weight_xpu is None
            or self._cached_weight_version != w_ver
            or self._cached_weight_xpu.device.type != "xpu"
            or self._cached_weight_xpu.dtype != torch.float16
            or not self._cached_weight_xpu.is_contiguous()
        ):
            self._cached_weight_xpu = (
                w.detach().to("xpu", dtype=torch.float16).contiguous()
            )
            self._cached_weight_version = w_ver

        if self.gemm.bias is None:
            self._cached_bias_xpu = None
            self._cached_bias_version = -1
        else:
            b = self.gemm.bias
            b_ver = int(b._version)
            if (
                self._cached_bias_xpu is None
                or self._cached_bias_version != b_ver
                or self._cached_bias_xpu.device.type != "xpu"
                or self._cached_bias_xpu.dtype != torch.float16
                or not self._cached_bias_xpu.is_contiguous()
            ):
                self._cached_bias_xpu = (
                    b.detach().to("xpu", dtype=torch.float16).contiguous()
                )
                self._cached_bias_version = b_ver

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to("xpu", dtype=torch.float16).contiguous()

        self._ensure_cached_params()
        return kernel_function(x, self._cached_weight_xpu, self._cached_bias_xpu)
