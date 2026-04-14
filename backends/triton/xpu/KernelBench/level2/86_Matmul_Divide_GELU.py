# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_autotune_configs():
    configs = []

    def add(bm, bn, bk, gsm, nw, ns, even_m, even_n, even_k):
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "GROUP_SIZE_M": gsm,
                    "EVEN_M": even_m,
                    "EVEN_N": even_n,
                    "EVEN_K": even_k,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )

    # Large-tile XPU-focused configs.
    # Include mandatory 256x256 / 32-warps variants and GROUP_SIZE_M=1 fallback.
    for gsm in (1, 4, 8):
        add(256, 256, 16, gsm, 32, 2, True, True, True)
        add(256, 256, 16, gsm, 32, 3, True, True, True)
        add(256, 256, 32, gsm, 32, 2, True, True, True)
        add(256, 256, 32, gsm, 32, 3, True, True, True)
        add(256, 256, 32, gsm, 32, 4, True, True, True)

    # Medium tiles for register-pressure / occupancy tradeoffs.
    for gsm in (1, 2, 4):
        add(128, 256, 32, gsm, 16, 2, True, True, True)
        add(128, 256, 64, gsm, 16, 2, True, True, True)
        add(256, 128, 32, gsm, 16, 2, True, True, True)
        add(256, 128, 64, gsm, 16, 2, True, True, True)
        add(128, 128, 32, gsm, 8, 2, True, True, True)
        add(128, 128, 64, gsm, 16, 2, True, True, True)
        add(128, 128, 32, gsm, 16, 3, True, True, True)

    # Smaller fallback tiles for less favorable shapes.
    for gsm in (1, 2, 4):
        add(64, 256, 32, gsm, 16, 2, True, True, True)
        add(64, 256, 64, gsm, 16, 2, True, True, True)
        add(256, 64, 32, gsm, 16, 2, True, True, True)
        add(128, 64, 32, gsm, 8, 2, True, True, True)
        add(128, 64, 64, gsm, 8, 2, True, True, True)
        add(64, 128, 32, gsm, 8, 2, True, True, True)
        add(64, 128, 64, gsm, 8, 2, True, True, True)
        add(64, 64, 32, gsm, 4, 2, True, True, True)
        add(64, 64, 64, gsm, 8, 2, True, True, True)

    # Boundary-safe variants for non-divisible shapes.
    add(256, 256, 16, 1, 32, 3, False, False, True)
    add(256, 256, 32, 1, 32, 3, False, False, True)
    add(128, 256, 32, 1, 16, 2, False, False, True)
    add(256, 128, 32, 1, 16, 2, False, False, True)
    add(128, 128, 32, 1, 8, 2, False, False, True)
    add(64, 128, 32, 1, 8, 2, False, False, True)
    add(64, 64, 32, 1, 4, 2, False, False, True)

    # A few K-boundary-safe configs too.
    add(256, 256, 32, 1, 32, 3, False, False, False)
    add(128, 128, 64, 1, 16, 2, False, False, False)
    add(64, 64, 64, 1, 8, 2, False, False, False)

    return configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _linear_div_gelu_kernel_packed_rhs(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_b,
    stride_om,
    stride_on,
    divisor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, n_start),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(k_tiles):
        if EVEN_M and EVEN_K:
            a = tl.load(x_bp)
        else:
            a = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")

        if EVEN_N and EVEN_K:
            b = tl.load(w_bp)
        else:
            b = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")

        acc = tl.dot(a, b, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = n_start + tl.arange(0, BLOCK_N)
    bias = tl.load(b_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)
    acc = (acc + bias[None, :]) / divisor

    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))
    y = y.to(tl.float16)

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(m_start, n_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if EVEN_M and EVEN_N:
        tl.store(out_bp, y)
    else:
        tl.store(out_bp, y, boundary_check=(0, 1))


def kernel_function(input, weight_packed, bias, divisor=10.0):
    """
    Fused Triton kernel for output = GELU((input @ weight_packed + bias) / divisor)
    input: [M, K] fp16 on XPU
    weight_packed: [K, N] fp16 on XPU
    bias: [N] fp16/fp32 on XPU
    """
    x_xpu = input.to(device="xpu", dtype=torch.float16).contiguous()
    w_xpu = weight_packed.to(device="xpu", dtype=torch.float16).contiguous()
    b_xpu = bias.to(device="xpu", dtype=torch.float16).contiguous()

    M, K = x_xpu.shape
    K_w, N = w_xpu.shape
    assert K == K_w and b_xpu.shape[0] == N

    out = torch.empty((M, N), device=x_xpu.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _linear_div_gelu_kernel_packed_rhs[grid](
        x_xpu,
        w_xpu,
        b_xpu,
        out,
        M,
        N,
        K,
        x_xpu.stride(0),
        x_xpu.stride(1),
        w_xpu.stride(0),
        w_xpu.stride(1),
        b_xpu.stride(0),
        out.stride(0),
        out.stride(1),
        float(divisor),
        grf_mode="auto",
    )
    return out


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size, dtype=torch.float16)]


def get_init_inputs():
    return [input_size, output_size, divisor]


class Model(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.input_size = input_size
        self.output_size = output_size
        self._packed_w = None
        self._bias_xpu = None

    def _lazy_init_xpu(self):
        if self._packed_w is None or self._bias_xpu is None:
            w = self.linear.weight.detach().to(device="xpu", dtype=torch.float16).contiguous()
            b = self.linear.bias.detach().to(device="xpu", dtype=torch.float16).contiguous()
            self._packed_w = w.t().contiguous()  # [K, N]
            self._bias_xpu = b

    def forward(self, x):
        self._lazy_init_xpu()
        x_xpu = x.to(device="xpu", dtype=torch.float16).contiguous()
        return kernel_function(x_xpu, self._packed_w, self._bias_xpu, self.divisor)