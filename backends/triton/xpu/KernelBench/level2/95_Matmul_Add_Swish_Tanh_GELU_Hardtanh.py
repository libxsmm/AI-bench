# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _sg1_autotune_configs():
    configs = []

    # Large XPU-oriented GEMM tiles, with GROUP_SIZE_M=1 fallback included.
    for cfg in [
        (256, 256, 16, 1, 32, 3),
        (256, 256, 32, 1, 32, 3),
        (256, 256, 16, 4, 32, 3),
        (256, 256, 32, 4, 32, 3),
        (256, 128, 16, 1, 32, 3),
        (256, 128, 32, 1, 32, 3),
        (256, 128, 16, 4, 32, 3),
        (256, 128, 32, 4, 32, 3),
        (128, 256, 16, 1, 32, 3),
        (128, 256, 32, 1, 32, 3),
        (128, 256, 16, 4, 32, 3),
        (128, 256, 32, 4, 32, 3),
        (256, 256, 32, 1, 16, 3),
        (256, 128, 32, 1, 16, 3),
        (128, 256, 32, 1, 16, 3),
        (128, 128, 32, 1, 16, 3),
        (128, 128, 64, 1, 16, 3),
        (128, 128, 32, 8, 16, 3),
        (128, 128, 64, 8, 16, 3),
        (64, 256, 32, 1, 16, 2),
        (64, 256, 64, 4, 16, 2),
        (256, 64, 32, 1, 16, 2),
        (128, 64, 32, 2, 8, 2),
        (128, 64, 64, 2, 8, 2),
        (64, 128, 32, 4, 8, 2),
        (64, 128, 64, 4, 8, 2),
        (64, 64, 32, 4, 8, 2),
        (64, 64, 64, 4, 8, 2),
    ]:
        bm, bn, bk, gs, nw, ns = cfg
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "GROUP_SIZE_M": gs,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


def _sg2_autotune_configs():
    configs = []
    # 1D pointwise/search space for bandwidth+SFU-heavy activation chain.
    for block_size, nw, ns in [
        (256, 4, 2),
        (256, 8, 2),
        (512, 4, 2),
        (512, 8, 2),
        (512, 8, 3),
        (512, 16, 2),
        (1024, 4, 2),
        (1024, 8, 2),
        (1024, 8, 3),
        (1024, 16, 2),
        (2048, 8, 2),
        (2048, 8, 3),
        (2048, 16, 2),
    ]:
        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE": block_size,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


# ----------------------------
# Subgraph 1: Triton GEMM + bias/add
# Keep GEMM separate from the heavy activation chain to avoid
# epilogue register-pressure collapse on Intel XPU.
# grf_mode remains a compiler option passed at launch, not in Config.
# ----------------------------
@triton.autotune(
    configs=_sg1_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _sg1_linear_add_kernel(
    x_ptr,
    wt_ptr,
    bias_ptr,
    addv_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wtk,
    stride_wtn,
    stride_ym,
    stride_yn,
    ADD_IS_PRECOMBINED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

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
    wt_bp = tl.make_block_ptr(
        base=wt_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, n_start),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_bp, boundary_check=(0, 1))
        b = tl.load(wt_bp, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        wt_bp = tl.advance(wt_bp, (BLOCK_K, 0))

    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if ADD_IS_PRECOMBINED:
        acc = acc + bias_vals[None, :]
    else:
        add_vals = tl.load(addv_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + (bias_vals + add_vals)[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(m_start, n_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(y_ptr.dtype.element_ty), boundary_check=(0, 1))


_SG1_EPI_CACHE = {}


def _get_fused_epilogue(bias: torch.Tensor, add_value: torch.Tensor) -> torch.Tensor:
    key = (
        int(bias.data_ptr()),
        int(add_value.data_ptr()),
        tuple(bias.shape),
        tuple(add_value.shape),
        str(bias.dtype),
        str(add_value.dtype),
        str(bias.device),
        str(add_value.device),
        int(getattr(bias, "_version", 0)),
        int(getattr(add_value, "_version", 0)),
    )
    cached = _SG1_EPI_CACHE.get(key)
    if cached is not None:
        return cached
    fused = (bias + add_value).contiguous()
    _SG1_EPI_CACHE.clear()
    _SG1_EPI_CACHE[key] = fused
    return fused


def _sg1_forward(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, add_value: torch.Tensor) -> torch.Tensor:
    assert x.device == weight_t.device == bias.device == add_value.device
    assert x.dtype == weight_t.dtype == bias.dtype == add_value.dtype
    assert x.dim() == 2 and weight_t.dim() == 2 and bias.dim() == 1 and add_value.dim() == 1

    x = x.contiguous()
    weight_t = weight_t.contiguous()
    bias = bias.contiguous()
    add_value = add_value.contiguous()

    B, I = x.shape
    Iw, O = weight_t.shape
    assert I == Iw
    y = torch.empty((B, O), device=x.device, dtype=x.dtype)

    add_is_precombined = int(bias.data_ptr() == add_value.data_ptr())

    if add_is_precombined:
        fused_bias = bias
    else:
        fused_bias = _get_fused_epilogue(bias, add_value)

    def grid(meta):
        return (triton.cdiv(B, meta["BLOCK_M"]) * triton.cdiv(O, meta["BLOCK_N"]),)

    _sg1_linear_add_kernel[grid](
        x,
        weight_t,
        fused_bias,
        add_value,
        y,
        B,
        O,
        I,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        y.stride(0),
        y.stride(1),
        ADD_IS_PRECOMBINED=add_is_precombined,
        grf_mode="auto",
    )
    return y


# ----------------------------
# Subgraph 2: Fused Activation Chain
# Kept as a standalone kernel because fully fusing into the GEMM
# epilogue is likely harmful on XPU due to GRF pressure.
# ----------------------------
@triton.jit
def _erf_approx(x):
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    ax = tl.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = a5
    poly = poly * t + a4
    poly = poly * t + a3
    poly = poly * t + a2
    poly = poly * t + a1
    poly = poly * t

    y = 1.0 - poly * tl.exp(-(ax * ax))
    sgn = tl.where(x >= 0, 1.0, -1.0)
    return sgn * y


@triton.autotune(
    configs=_sg2_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _sg2_act_chain_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    log2e = 1.4426950408889634
    inv_sqrt2 = 0.7071067811865476

    sig = 1.0 / (1.0 + tl.math.exp2((-xf) * log2e))
    sw = sig * xf

    sig2 = 1.0 / (1.0 + tl.math.exp2((-2.0 * sw) * log2e))
    th = 2.0 * sig2 - 1.0

    z = th * inv_sqrt2
    erfz = _erf_approx(z)
    gelu = 0.5 * th * (1.0 + erfz)

    clamped = tl.maximum(tl.minimum(gelu, max_val), min_val)
    tl.store(y_ptr + offs, clamped.to(x.dtype), mask=mask)


def _sg2_forward(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    assert x.device.type == "xpu"
    assert x.dtype in (torch.float16, torch.bfloat16)
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _sg2_act_chain_kernel[grid](
        x,
        y,
        n,
        float(min_val),
        float(max_val),
    )
    return y


# ----------------------------
# Top-Level Kernel Function
# Expects packed weight_t in shape [K, N]
# ----------------------------
def kernel_function(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, add_value: torch.Tensor) -> torch.Tensor:
    assert isinstance(x, torch.Tensor) and isinstance(weight_t, torch.Tensor)
    assert isinstance(bias, torch.Tensor) and isinstance(add_value, torch.Tensor)

    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if weight_t.device.type != "xpu" or weight_t.dtype != torch.float16:
        weight_t_xpu = weight_t.to("xpu", dtype=torch.float16).contiguous()
    else:
        weight_t_xpu = weight_t.contiguous()

    if bias.device.type != "xpu" or bias.dtype != torch.float16:
        bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bias_xpu = bias.contiguous()

    if add_value.device.type != "xpu" or add_value.dtype != torch.float16:
        addv_xpu = add_value.to("xpu", dtype=torch.float16).contiguous()
    else:
        addv_xpu = add_value.contiguous()

    y1 = _sg1_forward(x_xpu, weight_t_xpu, bias_xpu, addv_xpu)
    y2 = _sg2_forward(y1, -1.0, 1.0)
    return y2


# ----------------------------
# Reference Model for Testing
# ----------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, add_value_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.zeros(add_value_shape))

        self.register_buffer("_cached_weight_t_xpu", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("_cached_bias_xpu", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("_cached_add_value_xpu", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("_cached_fused_bias_add_xpu", torch.empty(0, dtype=torch.float16), persistent=False)
        self._cache_ready = False
        self._weight_version = -1
        self._bias_version = -1
        self._add_value_version = -1

    def _refresh_xpu_cache(self):
        weight_xpu = self.matmul.weight.detach().to("xpu", dtype=torch.float16).contiguous()
        bias_xpu = self.matmul.bias.detach().to("xpu", dtype=torch.float16).contiguous()
        add_value_xpu = self.add_value.detach().to("xpu", dtype=torch.float16).contiguous()

        self._cached_weight_t_xpu = weight_xpu.t().contiguous()
        self._cached_bias_xpu = bias_xpu
        self._cached_add_value_xpu = add_value_xpu
        self._cached_fused_bias_add_xpu = (bias_xpu + add_value_xpu).contiguous()
        self._weight_version = int(self.matmul.weight._version)
        self._bias_version = int(self.matmul.bias._version)
        self._add_value_version = int(self.add_value._version)
        self._cache_ready = True

    def _ensure_epilogue_cache_fresh(self):
        cur_weight_ver = int(self.matmul.weight._version)
        cur_bias_ver = int(self.matmul.bias._version)
        cur_add_ver = int(self.add_value._version)
        if (
            (cur_weight_ver != self._weight_version)
            or (cur_bias_ver != self._bias_version)
            or (cur_add_ver != self._add_value_version)
        ):
            self._refresh_xpu_cache()

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        if (
            (not self._cache_ready)
            or self._cached_weight_t_xpu.numel() == 0
            or self._cached_bias_xpu.numel() == 0
            or self._cached_add_value_xpu.numel() == 0
            or self._cached_fused_bias_add_xpu.numel() == 0
        ):
            self._refresh_xpu_cache()
        else:
            self._ensure_epilogue_cache_fresh()

        return kernel_function(
            x,
            self._cached_weight_t_xpu,
            self._cached_fused_bias_add_xpu,
            self._cached_fused_bias_add_xpu,
        )
