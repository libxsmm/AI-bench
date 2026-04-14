# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._xpu_params_ready = False
        self._packed_weight = None

    def _ensure_xpu_params(self):
        if not self._xpu_params_ready:
            w = self.linear.weight.data.to("xpu", dtype=torch.float16).contiguous()
            b = self.linear.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.linear.weight.data = w
            self.linear.bias.data = b
            self._packed_weight = w.t().contiguous()
            self._xpu_params_ready = True
        else:
            if self.linear.weight.data.device.type != "xpu" or self.linear.weight.data.dtype != torch.float16:
                self.linear.weight.data = self.linear.weight.data.to("xpu", dtype=torch.float16).contiguous()
                self._packed_weight = self.linear.weight.data.t().contiguous()
            elif not self.linear.weight.data.is_contiguous():
                self.linear.weight.data = self.linear.weight.data.contiguous()
                self._packed_weight = self.linear.weight.data.t().contiguous()

            if self.linear.bias.data.device.type != "xpu" or self.linear.bias.data.dtype != torch.float16:
                self.linear.bias.data = self.linear.bias.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.linear.bias.data.is_contiguous():
                self.linear.bias.data = self.linear.bias.data.contiguous()

            if self._packed_weight is None or self._packed_weight.device.type != "xpu" or not self._packed_weight.is_contiguous():
                self._packed_weight = self.linear.weight.data.t().contiguous()

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        self._ensure_xpu_params()
        return kernel_function(x, self._packed_weight, self.linear.bias)


batch_size = 1024
in_features, out_features = 4096, 4096


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features]


@triton.jit
def _softplus(x):
    log2e = 1.4426950408889634
    ax = tl.abs(x)
    return tl.log(1.0 + tl.math.exp2(-ax * log2e)) + tl.maximum(x, 0.0)


@triton.jit
def _tanh_from_softplus(sp):
    log2e = 1.4426950408889634
    two_sp = 2.0 * sp
    return 1.0 - 2.0 / (tl.math.exp2(two_sp * log2e) + 1.0)


@triton.jit
def _mish(x):
    sp = _softplus(x)
    t = _tanh_from_softplus(sp)
    return x * t


_gemm_configs = [
    # Large-tile XPU-oriented configs; includes required 256x256 / 32-warps cases.
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16,
            'GROUP_SIZE_M': 1, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=32, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=32, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=32, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=32, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 2, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 2, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=16, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 4, 'EVEN_M': True, 'EVEN_N': True, 'EVEN_K': True,
        },
        num_warps=8, num_stages=2,
    ),
    # Fallback arbitrary-shape configs.
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16,
            'GROUP_SIZE_M': 1, 'EVEN_M': False, 'EVEN_N': False, 'EVEN_K': False,
        },
        num_warps=32, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': False, 'EVEN_N': False, 'EVEN_K': False,
        },
        num_warps=32, num_stages=3,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1, 'EVEN_M': False, 'EVEN_N': False, 'EVEN_K': False,
        },
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 4, 'EVEN_M': False, 'EVEN_N': False, 'EVEN_K': False,
        },
        num_warps=8, num_stages=2,
    ),
]


@triton.autotune(configs=_gemm_configs, key=['M', 'N', 'K'])
@triton.jit
def _fused_linear_mish2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for _ in range(k_tiles):
        if EVEN_M and EVEN_K:
            a = tl.load(x_bp)
        else:
            a = tl.load(x_bp, boundary_check=(0, 1))

        if EVEN_N and EVEN_K:
            b = tl.load(w_bp)
        else:
            b = tl.load(w_bp, boundary_check=(0, 1))

        acc = tl.dot(a, b, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_SIZE_K))
        w_bp = tl.advance(w_bp, (BLOCK_SIZE_K, 0))

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = tl.max_contiguous(offs_n, BLOCK_SIZE_N)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    if EVEN_M and EVEN_N:
        tl.store(y_bp, acc.to(y_ptr.dtype.element_ty))
    else:
        tl.store(y_bp, acc.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32, num_stages=2),
    ],
    key=['numel'],
)
@triton.jit
def _mish2_kernel(
    x_ptr, y_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(offs, BLOCK_SIZE)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    log2e = 1.4426950408889634

    ax = tl.abs(x)
    sp = tl.log(1.0 + tl.math.exp2(-ax * log2e)) + tl.maximum(x, 0.0)
    t = 1.0 - 2.0 / (tl.math.exp2((2.0 * sp) * log2e) + 1.0)
    x = x * t

    ax = tl.abs(x)
    sp = tl.log(1.0 + tl.math.exp2(-ax * log2e)) + tl.maximum(x, 0.0)
    t = 1.0 - 2.0 / (tl.math.exp2((2.0 * sp) * log2e) + 1.0)
    x = x * t

    tl.store(y_ptr + offs, x.to(y_ptr.dtype.element_ty), mask=mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1

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

    M, Kx = x_xpu.shape
    Kw, N = w_xpu.shape
    assert Kx == Kw, "Incompatible shapes"
    assert N == b_xpu.shape[0]

    gemm_out = torch.empty((M, N), device=x_xpu.device, dtype=x_xpu.dtype)
    y = torch.empty((M, N), device=x_xpu.device, dtype=x_xpu.dtype)

    stride_xm, stride_xk = x_xpu.stride()
    stride_wk, stride_wn = w_xpu.stride()
    stride_gm, stride_gn = gemm_out.stride()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    _fused_linear_mish2_kernel[grid](
        x_xpu, w_xpu, b_xpu, gemm_out,
        M, N, Kx,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_gm, stride_gn,
        grf_mode="auto",
    )

    numel = gemm_out.numel()

    def elt_grid(meta):
        return (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    _mish2_kernel[elt_grid](gemm_out, y, numel)
    return y
