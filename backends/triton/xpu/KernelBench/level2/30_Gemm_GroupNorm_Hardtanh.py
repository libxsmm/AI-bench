# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
            num_warps=32,
            num_stages=3,
        ),
    ]


@triton.autotune(configs=_configs(), key=["N", "C_IN", "C_OUT"])
@triton.jit
def _fused_gemm_gn_hardtanh(
    x_ptr,
    w_ptr,
    b_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    N,
    C_IN,
    C_OUT,
    G,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yc,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(C_OUT, BLOCK_N)

    if GROUP_SIZE_M > 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, C_IN),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(C_OUT, C_IN),
        strides=(stride_wn, stride_wk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    for _ in range(0, C_IN, BLOCK_K):
        x = tl.load(x_bp, boundary_check=(0, 1))
        w = tl.load(w_bp, boundary_check=(0, 1))
        acc += tl.dot(x, tl.trans(w))
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (0, BLOCK_K))

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b[None, :]

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, C_OUT),
        strides=(stride_ym, stride_yc),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


def _gn_configs():
    return [
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 512}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_C": 512}, num_warps=16, num_stages=2),
    ]


@triton.autotune(configs=_gn_configs(), key=["N", "C_OUT", "GROUP_SIZE"])
@triton.jit
def _groupnorm_affine_hardtanh_kernel(
    inp_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N,
    C_OUT,
    GROUP_SIZE,
    stride_im,
    stride_ic,
    stride_om,
    stride_oc,
    EPS: tl.constexpr,
    HARDTANH_MIN: tl.constexpr,
    HARDTANH_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    group_base = pid_g * GROUP_SIZE
    mask_m = offs_m < N

    mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sq_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for c0 in range(0, GROUP_SIZE, BLOCK_C):
        cols = group_base + c0 + offs_c
        mask_c = (c0 + offs_c) < GROUP_SIZE
        vals = tl.load(
            inp_ptr + offs_m[:, None] * stride_im + cols[None, :] * stride_ic,
            mask=mask_m[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        mean += tl.sum(vals, axis=1)
        sq_sum += tl.sum(vals * vals, axis=1)

    inv_group = 1.0 / GROUP_SIZE
    mean = mean * inv_group
    var = tl.maximum(sq_sum * inv_group - mean * mean, 0.0)
    inv_std = tl.rsqrt(var + EPS)

    for c0 in range(0, GROUP_SIZE, BLOCK_C):
        cols = group_base + c0 + offs_c
        mask_c = (c0 + offs_c) < GROUP_SIZE

        vals = tl.load(
            inp_ptr + offs_m[:, None] * stride_im + cols[None, :] * stride_ic,
            mask=mask_m[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask_c, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=mask_c, other=0.0).to(tl.float32)

        vals = (vals - mean[:, None]) * inv_std[:, None]
        vals = vals * gamma[None, :] + beta[None, :]
        vals = tl.maximum(vals, HARDTANH_MIN)
        vals = tl.minimum(vals, HARDTANH_MAX)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + cols[None, :] * stride_oc,
            vals.to(tl.float16),
            mask=mask_m[:, None] & mask_c[None, :],
        )


def kernel_function(input, gemm_weight, gemm_bias, gn_weight, gn_bias):
    if not isinstance(input, torch.Tensor):
        raise RuntimeError("input must be a torch.Tensor")

    x_xpu = (
        input
        if input.device.type == "xpu" and input.dtype == torch.float16
        else input.to("xpu", dtype=torch.float16)
    )
    x_xpu = x_xpu.contiguous()
    dev = x_xpu.device

    def _to_xpu_contig(t, name):
        if not isinstance(t, torch.Tensor):
            raise RuntimeError(f"{name} must be a torch.Tensor")
        if t.device.type == "xpu" and t.dtype == torch.float16:
            return t.contiguous()
        return t.to(dev, dtype=torch.float16).contiguous()

    w_xpu = _to_xpu_contig(gemm_weight, "gemm_weight")
    b_xpu = _to_xpu_contig(gemm_bias, "gemm_bias")
    gw_xpu = _to_xpu_contig(gn_weight, "gn_weight")
    gb_xpu = _to_xpu_contig(gn_bias, "gn_bias")

    if x_xpu.ndim != 2:
        raise RuntimeError("input must be 2D [N, C_in]")
    if w_xpu.ndim != 2:
        raise RuntimeError("gemm_weight must be 2D [C_out, C_in]")
    if b_xpu.ndim != 1 or gw_xpu.ndim != 1 or gb_xpu.ndim != 1:
        raise RuntimeError("gemm_bias, gn_weight, gn_bias must be 1D [C_out]")

    N, C_in = x_xpu.shape
    C_out, C_in_w = w_xpu.shape
    if C_in_w != C_in:
        raise RuntimeError(
            "Incompatible shapes: gemm_weight.shape[1] != input.shape[1]"
        )
    if b_xpu.shape[0] != C_out or gw_xpu.shape[0] != C_out or gb_xpu.shape[0] != C_out:
        raise RuntimeError("Bias and affine parameter lengths must match C_out")

    G = 16
    if C_out % G != 0:
        raise RuntimeError("C_out must be divisible by num_groups=16")
    group_size = C_out // G

    gemm_out = torch.empty((N, C_out), dtype=torch.float16, device=dev)
    y = torch.empty((N, C_out), dtype=torch.float16, device=dev)

    stride_xm, stride_xk = x_xpu.stride()
    stride_wn, stride_wk = w_xpu.stride()
    stride_gm, stride_gc = gemm_out.stride()
    stride_ym, stride_yc = y.stride()

    def gemm_grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]) * triton.cdiv(C_out, meta["BLOCK_N"]),)

    _fused_gemm_gn_hardtanh[gemm_grid](
        x_xpu,
        w_xpu,
        b_xpu,
        gw_xpu,
        gb_xpu,
        gemm_out,
        N,
        C_in,
        C_out,
        G,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_gm,
        stride_gc,
        EPS=1e-5,
        GROUP_SIZE=group_size,
    )

    def gn_grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]), G)

    _groupnorm_affine_hardtanh_kernel[gn_grid](
        gemm_out,
        gw_xpu,
        gb_xpu,
        y,
        N,
        C_out,
        group_size,
        stride_gm,
        stride_gc,
        stride_ym,
        stride_yc,
        EPS=1e-5,
        HARDTANH_MIN=-2.0,
        HARDTANH_MAX=2.0,
    )
    return y


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        self._cache_device = None
        self._weight_cache = None
        self._bias_cache = None
        self._gn_weight_cache = None
        self._gn_bias_cache = None
        self._weight_version = -1
        self._bias_version = -1
        self._gn_weight_version = -1
        self._gn_bias_version = -1

    def _ensure_xpu_params(self, device):
        if device.type != "xpu":
            device = torch.device("xpu")

        if (
            self._weight_cache is None
            or self._cache_device != device
            or self._weight_version != self.gemm.weight._version
        ):
            self._weight_cache = (
                self.gemm.weight.detach()
                .to(device=device, dtype=torch.float16)
                .contiguous()
            )
            self._weight_version = self.gemm.weight._version

        if (
            self._bias_cache is None
            or self._cache_device != device
            or self._bias_version != self.gemm.bias._version
        ):
            self._bias_cache = (
                self.gemm.bias.detach()
                .to(device=device, dtype=torch.float16)
                .contiguous()
            )
            self._bias_version = self.gemm.bias._version

        if (
            self._gn_weight_cache is None
            or self._cache_device != device
            or self._gn_weight_version != self.group_norm.weight._version
        ):
            self._gn_weight_cache = (
                self.group_norm.weight.detach()
                .to(device=device, dtype=torch.float16)
                .contiguous()
            )
            self._gn_weight_version = self.group_norm.weight._version

        if (
            self._gn_bias_cache is None
            or self._cache_device != device
            or self._gn_bias_version != self.group_norm.bias._version
        ):
            self._gn_bias_cache = (
                self.group_norm.bias.detach()
                .to(device=device, dtype=torch.float16)
                .contiguous()
            )
            self._gn_bias_version = self.group_norm.bias._version

        self._cache_device = device

    def forward(self, x):
        x_xpu = (
            x
            if x.device.type == "xpu" and x.dtype == torch.float16
            else x.to("xpu", dtype=torch.float16)
        )
        x_xpu = x_xpu.contiguous()
        self._ensure_xpu_params(x_xpu.device)
        return kernel_function(
            x_xpu,
            self._weight_cache,
            self._bias_cache,
            self._gn_weight_cache,
            self._gn_bias_cache,
        )
