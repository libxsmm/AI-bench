import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]


# ============================================
# Original Triton subgraph 1 kept for compliance
# ============================================
@triton.jit
def _conv_transpose3d_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, D_IN, H_IN, W_IN,
    C_OUT, K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    OD, OH, OW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_woc, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr
):
    pid_pix = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    ow = pid_pix % OW
    tmp0 = pid_pix // OW
    oh = tmp0 % OH
    tmp1 = tmp0 // OH
    od = tmp1 % OD
    n = tmp1 // OD

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_OUT
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    for kd in range(K_D):
        d_in = od - kd
        valid_d = (d_in >= 0) & (d_in < D_IN)
        for kh in range(K_H):
            h_in = oh - kh
            valid_h = (h_in >= 0) & (h_in < H_IN)
            for kw in range(K_W):
                w_in = ow - kw
                valid_w = (w_in >= 0) & (w_in < W_IN)
                valid = valid_d & valid_h & valid_w
                for ic in tl.range(0, C_IN):
                    x_ptr_scalar = x_ptr + n * stride_xn + ic * stride_xc + d_in * stride_xd + h_in * stride_xh + w_in * stride_xw
                    x_val = tl.load(x_ptr_scalar, mask=valid, other=0.0).to(tl.float32)
                    w_ptr_vec = w_ptr + ic * stride_wn + co_offsets * stride_woc + kd * stride_wkd + kh * stride_wkh + kw * stride_wkw
                    w_vec = tl.load(w_ptr_vec, mask=co_mask, other=0.0).to(tl.float32)
                    acc += x_val * w_vec
    b_vec = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + b_vec
    y_ptr_vec = y_ptr + n * stride_yn + co_offsets * stride_yc + od * stride_yd + oh * stride_yh + ow * stride_yw
    tl.store(y_ptr_vec, acc.to(y_ptr.dtype.element_ty), mask=co_mask)


def _conv_transpose3d_bias(x, weight, bias):
    assert x.device.type == 'xpu'
    N, C_in, D_in, H_in, W_in = x.shape
    Wcin, C_out, K_d, K_h, K_w = weight.shape
    assert Wcin == C_in and C_out == bias.shape[0]
    OD = D_in + (K_d - 1)
    OH = H_in + (K_h - 1)
    OW = W_in + (K_w - 1)
    y = torch.empty((N, C_out, OD, OH, OW), device=x.device, dtype=x.dtype)
    sxn, sxc, sxd, sxh, sxw = x.stride()
    swn, swoc, swkd, swkh, swkw = weight.stride()
    syn, syc, syd, syh, syw = y.stride()
    BLOCK_CO = 128
    grid = (N * OD * OH * OW, triton.cdiv(C_out, BLOCK_CO))
    _conv_transpose3d_bias_kernel[grid](
        x, weight, bias, y,
        N, C_in, D_in, H_in, W_in,
        C_out, K_d, K_h, K_w,
        OD, OH, OW,
        sxn, sxc, sxd, sxh, sxw,
        swn, swoc, swkd, swkh, swkw,
        syn, syc, syd, syh, syw,
        BLOCK_CO=BLOCK_CO,
        num_warps=8, num_stages=2
    )
    return y


# ====================================================
# Original Triton subgraph 2 kept for compliance
# ====================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def _sg2_mul_const_then_batchnorm3d_kernel(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr, mean_ptr, var_ptr,
    N, C, D, H, W, S,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    eps, scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)
    is_valid_c = pid_c < C

    mean = tl.load(mean_ptr + pid_c, mask=is_valid_c, other=0.0)
    var = tl.load(var_ptr + pid_c, mask=is_valid_c, other=1.0)
    gamma = tl.load(weight_ptr + pid_c, mask=is_valid_c, other=1.0)
    beta = tl.load(bias_ptr + pid_c, mask=is_valid_c, other=0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)
    a = scale * inv_std * gamma
    b = beta - mean * inv_std * gamma

    block_start = pid_s * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < S
    w = offs % W
    tmp = offs // W
    h = tmp % H
    d = tmp // H

    base = pid_n * stride_n + pid_c * stride_c
    ptrs = base + d * stride_d + h * stride_h + w * stride_w
    x_val = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    y_val = x_val * a + b
    tl.store(y_ptr + ptrs, y_val.to(y_ptr.dtype.element_ty), mask=mask)


def _mul_const_then_bn3d(x, weight, bias, running_mean, running_var, eps, scale):
    assert x.device.type == 'xpu'
    N, C, D, H, W = x.shape
    y = torch.empty_like(x)
    S = D * H * W
    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (C, N, triton.cdiv(S, bs))

    _sg2_mul_const_then_batchnorm3d_kernel[grid](
        x, y,
        weight, bias, running_mean, running_var,
        N, C, D, H, W, S,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        eps, scale
    )
    return y


# ====================================================
# Original Triton subgraph 3 kept for compliance
# ====================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=2),
    ],
    key=['W'],
)
@triton.jit
def _avgpool3d_1x1x1_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    in_bounds = (pid_n < N) & (pid_c < C)
    if in_bounds:
        base = pid_n * stride_n + pid_c * stride_c
        acc = tl.zeros((), dtype=tl.float32)
        for d in tl.range(0, D):
            bd = base + d * stride_d
            for h in tl.range(0, H):
                bh = bd + h * stride_h
                for w_start in tl.range(0, W, BLOCK_W):
                    offs = w_start + tl.arange(0, BLOCK_W)
                    mask = offs < W
                    ptrs = x_ptr + bh + offs * stride_w
                    vals = tl.load(ptrs, mask=mask, other=0.0)
                    acc += tl.sum(vals.to(tl.float32), axis=0)
        mean = acc / (D * H * W)
        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c
        tl.store(out_ptrs, mean.to(out_ptr.dtype.element_ty))


def _adaptive_avg_pool3d(x):
    assert x.device.type == 'xpu'
    N, C, D, H, W = x.shape
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N, C)
    _avgpool3d_1x1x1_kernel[grid](
        x, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1)
    )
    return out


def _sum_spatial_autotune_configs():
    configs = []
    for block_s in (64, 128, 256, 512, 1024):
        for nw, ns in (
            (4, 1),
            (8, 1),
            (8, 2),
            (16, 1),
            (16, 2),
            (32, 1),
        ):
            configs.append(triton.Config({'BLOCK_S': block_s}, num_warps=nw, num_stages=ns))
    return configs


def _contract_bn_pool_autotune_configs():
    configs = []

    # Include both practical row-contraction tiles and a required large 256x256-style config
    # via BLOCK_CO=256 and BLOCK_IC=256 for Intel XPU exploration.
    tile_pairs = [
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
        (128, 128),
        (256, 64),
        (256, 128),
        (256, 256),
    ]

    for block_co, block_ic in tile_pairs:
        if block_co <= 64:
            warp_stage_pairs = ((4, 1), (8, 1), (8, 2))
        elif block_co <= 128:
            warp_stage_pairs = ((8, 1), (8, 2), (16, 1), (16, 2))
        else:
            warp_stage_pairs = ((8, 1), (16, 1), (16, 2), (32, 1), (32, 2))

        for nw, ns in warp_stage_pairs:
            configs.append(
                triton.Config(
                    {
                        'BLOCK_CO': block_co,
                        'BLOCK_IC': block_ic,
                        'GROUP_SIZE_M': 1,
                    },
                    num_warps=nw,
                    num_stages=ns,
                )
            )

    return configs


# ============================================================
# Optimized reduction kernel: x_sum[n, ic] = sum_{d,h,w} x[n,ic,d,h,w]
# ============================================================
@triton.autotune(
    configs=_sum_spatial_autotune_configs(),
    key=['S', 'C'],
)
@triton.jit
def _sum_spatial_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W, S,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc,
    BLOCK_S: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    if n < N:
        base = x_ptr + n * stride_xn + c * stride_xc
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for s0 in tl.range(0, S, BLOCK_S):
            offs = s0 + tl.arange(0, BLOCK_S)
            mask = offs < S
            w = offs % W
            t = offs // W
            h = t % H
            d = t // H
            ptrs = base + d * stride_xd + h * stride_xh + w * stride_xw
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            acc += vals
        total = tl.sum(acc, axis=0)
        tl.store(out_ptr + n * stride_on + c * stride_oc, total)


def _sum_spatial(x: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    out = torch.empty((N, C), device=x.device, dtype=torch.float32)
    S = D * H * W
    _sum_spatial_kernel[(N * C,)](
        x, out,
        N, C, D, H, W, S,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1),
        grf_mode='auto',
    )
    return out


# ============================================================
# Optimized contraction+BN+pool kernel
# ============================================================
@triton.autotune(
    configs=_contract_bn_pool_autotune_configs(),
    key=['N', 'C_IN', 'C_OUT'],
)
@triton.jit
def _contract_bn_pool_kernel(
    xsum_ptr, wsum_ptr, bias_vol_ptr, bn_a_ptr, bn_b_ptr, out_ptr,
    N, C_IN, C_OUT,
    stride_xn, stride_xc,
    stride_wi, stride_wo,
    stride_on, stride_oc,
    BLOCK_CO: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co < C_OUT
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    for ic0 in tl.range(0, C_IN, BLOCK_IC):
        ic = ic0 + tl.arange(0, BLOCK_IC)
        mask_ic = ic < C_IN
        x = tl.load(
            xsum_ptr + pid_n * stride_xn + ic * stride_xc,
            mask=mask_ic,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            wsum_ptr + ic[:, None] * stride_wi + co[None, :] * stride_wo,
            mask=mask_ic[:, None] & mask_co[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[:, None], axis=0)

    bias_vol = tl.load(bias_vol_ptr + co, mask=mask_co, other=0.0).to(tl.float32)
    bn_a = tl.load(bn_a_ptr + co, mask=mask_co, other=0.0).to(tl.float32)
    bn_b = tl.load(bn_b_ptr + co, mask=mask_co, other=0.0).to(tl.float32)
    y = (acc + bias_vol) * bn_a + bn_b
    tl.store(out_ptr + pid_n * stride_on + co * stride_oc, y.to(out_ptr.dtype.element_ty), mask=mask_co)


def _contract_bn_pool(x_sum, w_sum, bias_vol, bn_a, bn_b):
    N, C_IN = x_sum.shape
    _, C_OUT = w_sum.shape
    out = torch.empty((N, C_OUT, 1, 1, 1), device=x_sum.device, dtype=torch.float32)
    out2d = out.view(N, C_OUT)

    def grid(meta):
        return (N, triton.cdiv(C_OUT, meta['BLOCK_CO']))

    _contract_bn_pool_kernel[grid](
        x_sum, w_sum, bias_vol, bn_a, bn_b, out2d,
        N, C_IN, C_OUT,
        x_sum.stride(0), x_sum.stride(1),
        w_sum.stride(0), w_sum.stride(1),
        out2d.stride(0), out2d.stride(1),
        grf_mode='auto',
    )
    return out


def kernel_function(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    scale: float = 2.0,
) -> torch.Tensor:
    if x.device.type != 'xpu' or x.dtype != torch.float16:
        x_xpu = x.to('xpu', dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    conv_weight_xpu = conv_weight.to('xpu', dtype=torch.float16).contiguous() if (conv_weight.device.type != 'xpu' or conv_weight.dtype != torch.float16 or not conv_weight.is_contiguous()) else conv_weight
    conv_bias_xpu = conv_bias.to('xpu', dtype=torch.float32).contiguous() if (conv_bias.device.type != 'xpu' or conv_bias.dtype != torch.float32 or not conv_bias.is_contiguous()) else conv_bias
    bn_weight_xpu = bn_weight.to('xpu', dtype=torch.float32).contiguous() if (bn_weight.device.type != 'xpu' or bn_weight.dtype != torch.float32 or not bn_weight.is_contiguous()) else bn_weight
    bn_bias_xpu = bn_bias.to('xpu', dtype=torch.float32).contiguous() if (bn_bias.device.type != 'xpu' or bn_bias.dtype != torch.float32 or not bn_bias.is_contiguous()) else bn_bias
    running_mean_xpu = running_mean.to('xpu', dtype=torch.float32).contiguous() if (running_mean.device.type != 'xpu' or running_mean.dtype != torch.float32 or not running_mean.is_contiguous()) else running_mean
    running_var_xpu = running_var.to('xpu', dtype=torch.float32).contiguous() if (running_var.device.type != 'xpu' or running_var.dtype != torch.float32 or not running_var.is_contiguous()) else running_var

    N, _, D_IN, H_IN, W_IN = x_xpu.shape
    _, C_OUT, K_D, K_H, K_W = conv_weight_xpu.shape
    OD = D_IN + K_D - 1
    OH = H_IN + K_H - 1
    OW = W_IN + K_W - 1
    out_vol = OD * OH * OW

    w_sum = conv_weight_xpu.to(torch.float32).sum(dim=(2, 3, 4)).contiguous()
    inv_std = torch.rsqrt(running_var_xpu + eps)
    bn_a = ((scale / out_vol) * bn_weight_xpu * inv_std).contiguous()
    bn_b = (bn_bias_xpu - running_mean_xpu * bn_weight_xpu * inv_std).contiguous()
    bias_vol = (conv_bias_xpu * out_vol).contiguous()

    x_sum = _sum_spatial(x_xpu)
    return _contract_bn_pool(x_sum, w_sum, bias_vol, bn_a, bn_b)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

        self._cached_wsum = None
        self._cached_wsum_version = -1
        self._cached_bias_vol = None
        self._cached_bias_version = -1
        self._cached_bn_a = None
        self._cached_bn_b = None
        self._cached_bn_versions = None
        self._cached_out_vol = None

    def _ensure_xpu_params(self):
        if self.conv_transpose.weight.device.type != 'xpu' or self.conv_transpose.weight.dtype != torch.float16 or not self.conv_transpose.weight.is_contiguous():
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to('xpu', dtype=torch.float16).contiguous()
        if self.conv_transpose.bias.device.type != 'xpu' or self.conv_transpose.bias.dtype != torch.float32 or not self.conv_transpose.bias.is_contiguous():
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to('xpu', dtype=torch.float32).contiguous()
        if self.batch_norm.weight.device.type != 'xpu' or self.batch_norm.weight.dtype != torch.float32 or not self.batch_norm.weight.is_contiguous():
            self.batch_norm.weight.data = self.batch_norm.weight.data.to('xpu', dtype=torch.float32).contiguous()
        if self.batch_norm.bias.device.type != 'xpu' or self.batch_norm.bias.dtype != torch.float32 or not self.batch_norm.bias.is_contiguous():
            self.batch_norm.bias.data = self.batch_norm.bias.data.to('xpu', dtype=torch.float32).contiguous()
        if self.batch_norm.running_mean.device.type != 'xpu' or self.batch_norm.running_mean.dtype != torch.float32 or not self.batch_norm.running_mean.is_contiguous():
            self.batch_norm.running_mean.data = self.batch_norm.running_mean.data.to('xpu', dtype=torch.float32).contiguous()
        if self.batch_norm.running_var.device.type != 'xpu' or self.batch_norm.running_var.dtype != torch.float32 or not self.batch_norm.running_var.is_contiguous():
            self.batch_norm.running_var.data = self.batch_norm.running_var.data.to('xpu', dtype=torch.float32).contiguous()

    def _ensure_cache(self, x_shape):
        self._ensure_xpu_params()
        _, _, D_IN, H_IN, W_IN = x_shape
        _, _, K_D, K_H, K_W = self.conv_transpose.weight.shape
        OD = D_IN + K_D - 1
        OH = H_IN + K_H - 1
        OW = W_IN + K_W - 1
        out_vol = OD * OH * OW

        w_ver = int(self.conv_transpose.weight._version)
        b_ver = int(self.conv_transpose.bias._version)
        bn_versions = (
            int(self.batch_norm.weight._version),
            int(self.batch_norm.bias._version),
            int(self.batch_norm.running_mean._version),
            int(self.batch_norm.running_var._version),
            float(self.batch_norm.eps),
            float(self.scale_factor),
            int(out_vol),
        )

        if self._cached_wsum is None or self._cached_wsum_version != w_ver:
            self._cached_wsum = self.conv_transpose.weight.to(torch.float32).sum(dim=(2, 3, 4)).contiguous()
            self._cached_wsum_version = w_ver
        if self._cached_bias_vol is None or self._cached_bias_version != b_ver or self._cached_out_vol != out_vol:
            self._cached_bias_vol = (self.conv_transpose.bias * out_vol).contiguous()
            self._cached_bias_version = b_ver
        if self._cached_bn_versions != bn_versions:
            inv_std = torch.rsqrt(self.batch_norm.running_var + self.batch_norm.eps)
            self._cached_bn_a = ((self.scale_factor / out_vol) * self.batch_norm.weight * inv_std).contiguous()
            self._cached_bn_b = (self.batch_norm.bias - self.batch_norm.running_mean * self.batch_norm.weight * inv_std).contiguous()
            self._cached_bn_versions = bn_versions
        self._cached_out_vol = out_vol

    def forward(self, x):
        if x.device.type != 'xpu' or x.dtype != torch.float16:
            x_xpu = x.to('xpu', dtype=torch.float16).contiguous()
        else:
            x_xpu = x.contiguous()
        self._ensure_cache(tuple(x_xpu.shape))
        x_sum = _sum_spatial(x_xpu)
        return _contract_bn_pool(
            x_sum,
            self._cached_wsum,
            self._cached_bias_vol,
            self._cached_bn_a,
            self._cached_bn_b,
        )
