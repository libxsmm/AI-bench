# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_mul_bn_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    scale_ptr,
    mean_ptr,
    var_ptr,
    bn_w_ptr,
    bn_b_ptr,
    out_ptr,
    M,
    N,
    K,
    eps,
    stride_x_m,
    stride_x_k,
    stride_w_n,
    stride_w_k,
    stride_b_n,
    stride_scale_n,
    stride_mean_n,
    stride_var_n,
    stride_bn_w_n,
    stride_bn_b_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    zero = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tl.store(out_ptrs, zero, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _precompute_epilogue_kernel(
    b_ptr,
    scale_ptr,
    mean_ptr,
    var_ptr,
    bn_w_ptr,
    bn_b_ptr,
    fused_mul_ptr,
    fused_add_ptr,
    N,
    eps,
    stride_b_n,
    stride_scale_n,
    stride_mean_n,
    stride_var_n,
    stride_bn_w_n,
    stride_bn_b_n,
    stride_fused_mul_n,
    stride_fused_add_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    bias = tl.load(b_ptr + offs_n * stride_b_n, mask=mask_n, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + offs_n * stride_scale_n, mask=mask_n, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_n * stride_mean_n, mask=mask_n, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_n * stride_var_n, mask=mask_n, other=0.0).to(tl.float32)
    gamma = tl.load(bn_w_ptr + offs_n * stride_bn_w_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(bn_b_ptr + offs_n * stride_bn_b_n, mask=mask_n, other=0.0).to(tl.float32)

    inv_std = tl.rsqrt(var + eps)
    gain = gamma * inv_std
    scaled_bias = bias * scale
    mul = scale * gain
    add = beta + (scaled_bias - mean) * gain

    tl.store(fused_mul_ptr + offs_n * stride_fused_mul_n, mul, mask=mask_n)
    tl.store(fused_add_ptr + offs_n * stride_fused_add_n, add, mask=mask_n)


@triton.jit
def _epilogue_apply_kernel(
    y_ptr,
    fused_mul_ptr,
    fused_add_ptr,
    out_ptr,
    M,
    N,
    stride_y_m,
    stride_y_n,
    stride_fused_mul_n,
    stride_fused_add_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)

    mul = tl.load(fused_mul_ptr + offs_n * stride_fused_mul_n, mask=mask_n, other=0.0).to(tl.float32)
    add = tl.load(fused_add_ptr + offs_n * stride_fused_add_n, mask=mask_n, other=0.0).to(tl.float32)

    out = y * mul[None, :] + add[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, out.to(tl.float16), mask=mask)


@triton.jit
def _copy_fp16_kernel(
    src_ptr,
    dst_ptr,
    M,
    N,
    stride_src_m,
    stride_src_n,
    stride_dst_m,
    stride_dst_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    src_ptrs = src_ptr + offs_m[:, None] * stride_src_m + offs_n[None, :] * stride_src_n
    dst_ptrs = dst_ptr + offs_m[:, None] * stride_dst_m + offs_n[None, :] * stride_dst_n
    x = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, x, mask=mask)


def _to_xpu_contig(t, dtype):
    if t.device.type != "xpu" or t.dtype != dtype or not t.is_contiguous():
        return t.to("xpu", dtype=dtype).contiguous()
    return t


def _tensor_cache_key(t):
    return (
        t.data_ptr(),
        int(getattr(t, "_version", 0)),
        str(t.device),
        t.dtype,
        tuple(t.shape),
        tuple(t.stride()),
    )


def _epilogue_only(y, fused_mul, fused_add):
    M, N = y.shape
    out = torch.empty((M, N), device=y.device, dtype=torch.float16)

    # Keep the standalone Triton epilogue rather than replacing the GEMM path.
    # This stage applies only safe fusion-adjacent tuning.
    BLOCK_M = 64
    BLOCK_N = 256
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _epilogue_apply_kernel[grid](
        y,
        fused_mul,
        fused_add,
        out,
        M,
        N,
        y.stride(0),
        y.stride(1),
        fused_mul.stride(0),
        fused_add.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return out


def kernel_function(
    x,
    w,
    b,
    scale,
    running_mean,
    running_var,
    bn_weight,
    bn_bias,
    eps=1e-5,
    fused_mul=None,
    fused_add=None,
):
    x_xpu = _to_xpu_contig(x, torch.float16)
    w_xpu = _to_xpu_contig(w, torch.float16)
    b_xpu = _to_xpu_contig(b, torch.float16)

    if fused_mul is None or fused_add is None:
        scale_xpu = _to_xpu_contig(scale, torch.float16)
        running_mean_xpu = _to_xpu_contig(running_mean, torch.float32)
        running_var_xpu = _to_xpu_contig(running_var, torch.float32)
        bn_weight_xpu = _to_xpu_contig(bn_weight, torch.float16)
        bn_bias_xpu = _to_xpu_contig(bn_bias, torch.float16)

        n = w_xpu.shape[0]
        fused_mul_xpu = torch.empty((n,), device=x_xpu.device, dtype=torch.float32)
        fused_add_xpu = torch.empty((n,), device=x_xpu.device, dtype=torch.float32)
        grid_aff = (triton.cdiv(n, 256),)
        _precompute_epilogue_kernel[grid_aff](
            b_xpu,
            scale_xpu,
            running_mean_xpu,
            running_var_xpu,
            bn_weight_xpu,
            bn_bias_xpu,
            fused_mul_xpu,
            fused_add_xpu,
            n,
            eps,
            b_xpu.stride(0),
            scale_xpu.stride(0),
            running_mean_xpu.stride(0),
            running_var_xpu.stride(0),
            bn_weight_xpu.stride(0),
            bn_bias_xpu.stride(0),
            fused_mul_xpu.stride(0),
            fused_add_xpu.stride(0),
            BLOCK_N=256,
            num_warps=4,
            num_stages=1,
        )
    else:
        fused_mul_xpu = _to_xpu_contig(fused_mul, torch.float32)
        fused_add_xpu = _to_xpu_contig(fused_add, torch.float32)

    y = torch.mm(x_xpu, w_xpu.transpose(0, 1))
    return _epilogue_only(y, fused_mul_xpu, fused_add_xpu)


batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self._cached_fused_mul = None
        self._cached_fused_add = None
        self._cached_affine_key = None
        self._xpu_weight = None
        self._xpu_weight_key = None

    def _ensure_xpu_params(self):
        self.linear.weight.data = _to_xpu_contig(self.linear.weight.data, torch.float16)
        self.linear.bias.data = _to_xpu_contig(self.linear.bias.data, torch.float16)
        self.scale.data = _to_xpu_contig(self.scale.data, torch.float16)
        self.bn.weight.data = _to_xpu_contig(self.bn.weight.data, torch.float16)
        self.bn.bias.data = _to_xpu_contig(self.bn.bias.data, torch.float16)
        self.bn.running_mean.data = _to_xpu_contig(self.bn.running_mean.data, torch.float32)
        self.bn.running_var.data = _to_xpu_contig(self.bn.running_var.data, torch.float32)

    def _ensure_cached_weight(self):
        self._ensure_xpu_params()
        key = _tensor_cache_key(self.linear.weight)
        if self._xpu_weight is None or self._xpu_weight_key != key:
            self._xpu_weight = self.linear.weight
            self._xpu_weight_key = key

    def _ensure_cached_affine(self):
        self._ensure_xpu_params()
        key = (
            _tensor_cache_key(self.linear.bias),
            _tensor_cache_key(self.scale),
            _tensor_cache_key(self.bn.weight),
            _tensor_cache_key(self.bn.bias),
            _tensor_cache_key(self.bn.running_mean),
            _tensor_cache_key(self.bn.running_var),
            float(self.eps),
        )
        if key != self._cached_affine_key or self._cached_fused_mul is None or self._cached_fused_add is None:
            n = self.scale.numel()
            self._cached_fused_mul = torch.empty((n,), device="xpu", dtype=torch.float32)
            self._cached_fused_add = torch.empty((n,), device="xpu", dtype=torch.float32)
            BLOCK_N = 256
            grid = (triton.cdiv(n, BLOCK_N),)
            _precompute_epilogue_kernel[grid](
                self.linear.bias,
                self.scale,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self._cached_fused_mul,
                self._cached_fused_add,
                n,
                self.eps,
                self.linear.bias.stride(0),
                self.scale.stride(0),
                self.bn.running_mean.stride(0),
                self.bn.running_var.stride(0),
                self.bn.weight.stride(0),
                self.bn.bias.stride(0),
                self._cached_fused_mul.stride(0),
                self._cached_fused_add.stride(0),
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=1,
            )
            self._cached_affine_key = key

    def forward(self, x):
        self._ensure_cached_weight()
        self._ensure_cached_affine()
        return kernel_function(
            x,
            self._xpu_weight,
            self.linear.bias,
            self.scale,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.eps,
            fused_mul=self._cached_fused_mul,
            fused_add=self._cached_fused_add,
        )
