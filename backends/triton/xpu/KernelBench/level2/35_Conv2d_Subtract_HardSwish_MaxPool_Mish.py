# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv2d_xpu_autotune_configs():
    configs = []

    # Map GEMM-like recommendations onto this conv kernel:
    # M ~ output channels tile, N ~ output width tile, K ~ input channels reduction tile
    base_tiles = [
        (32, 32, 16, 1),
        (32, 64, 16, 1),
        (64, 32, 16, 1),
        (64, 64, 16, 1),
        (64, 64, 32, 1),
        (64, 128, 16, 1),
        (64, 128, 32, 1),
        (128, 64, 16, 1),
        (128, 64, 32, 1),
        (128, 128, 16, 1),
        (128, 128, 32, 1),
        (256, 128, 16, 1),
        (128, 256, 16, 1),
        (256, 256, 16, 1),  # required large-tile / 32-warp candidate
    ]

    for block_oc, block_ow, block_ic, group_size_m in base_tiles:
        for num_warps in (4, 8, 16, 32):
            for num_stages in (1, 2, 3):
                if block_oc == 256 and block_ow == 256 and num_warps not in (16, 32):
                    continue
                if block_ic == 32 and num_warps == 4 and num_stages == 3:
                    continue
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_OC": block_oc,
                            "BLOCK_OW": block_ow,
                            "BLOCK_IC": block_ic,
                            "GROUP_SIZE_M": group_size_m,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    return configs


def _pointwise_xpu_autotune_configs():
    configs = []
    for block_size in (128, 256, 512, 1024):
        for num_warps in (4, 8, 16, 32):
            for num_stages in (1, 2, 3, 4):
                if block_size == 1024 and num_warps == 4:
                    continue
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_SIZE": block_size,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_conv2d_xpu_autotune_configs(),
    key=["N", "C_in", "C_out", "H_out", "W_out"],
)
@triton.jit
def _conv2d_nchw_3x3_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W, C_out,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    USE_BF16: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_oc = tl.program_id(2)

    num_pid_m = N * H_out
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = tl.minimum(first_pid_m + (pid_m % GROUP_SIZE_M), num_pid_m - 1)

    oh = pid_m % H_out
    n = pid_m // H_out
    n64 = n.to(tl.int64)
    oh64 = oh.to(tl.int64)

    oc_start = pid_oc * BLOCK_OC
    ow_start = pid_ow * BLOCK_OW

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    ow_offsets = ow_start + tl.arange(0, BLOCK_OW)
    oc_mask = oc_offsets < C_out
    ow_mask = ow_offsets < W_out
    oc_offsets64 = oc_offsets.to(tl.int64)
    ow_offsets64 = ow_offsets.to(tl.int64)

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    x_batch_off = n64 * stride_xn
    y_batch_off = n64 * stride_yn

    for ic_start in range(0, C_in, BLOCK_IC):
        ic_offsets = ic_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offsets < C_in
        ic_offsets64 = ic_offsets.to(tl.int64)

        for kh in range(3):
            ih64 = oh64 + kh
            for kw in range(3):
                iw64 = ow_offsets64 + kw

                x_ptrs = (
                    x_ptr
                    + x_batch_off
                    + ic_offsets64[:, None] * stride_xc
                    + ih64 * stride_xh
                    + iw64[None, :] * stride_xw
                )
                x_mask = ic_mask[:, None] & ow_mask[None, :]
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_ptrs = (
                    w_ptr
                    + oc_offsets64[:, None] * stride_wo
                    + ic_offsets64[None, :] * stride_wi
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc = tl.dot(w_tile.to(tl.float32), x_tile.to(tl.float32), acc)

    b_vals = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals[:, None]

    y_ptrs = (
        y_ptr
        + y_batch_off
        + oc_offsets64[:, None] * stride_yc
        + oh64 * stride_yh
        + ow_offsets64[None, :] * stride_yw
    )
    y_mask = oc_mask[:, None] & ow_mask[None, :]

    y_vals = acc.to(tl.bfloat16) if USE_BF16 else acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y_vals, mask=y_mask)


@triton.autotune(
    configs=_pointwise_xpu_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _hardswish_sub_kernel(
    x_ptr, y_ptr, n_elements, subtract_value,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    z = x_f32 - subtract_value
    t = tl.minimum(tl.maximum(z + 3.0, 0.0), 6.0)
    y_f32 = z * t * (1.0 / 6.0)
    tl.store(y_ptr + offs, y_f32.to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=_pointwise_xpu_autotune_configs(),
    key=["N", "C", "OUT_H", "OUT_W"],
)
@triton.jit
def _maxpool2d_mish_kernel(
    x_ptr, y_ptr,
    N, C, H, W, OUT_H, OUT_W,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_sp = tl.program_id(1)
    offs = pid_sp * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (OUT_H * OUT_W)

    n = pid_nc // C
    c = pid_nc % C
    n64 = n.to(tl.int64)
    c64 = c.to(tl.int64)

    oh = offs // OUT_W
    ow = offs - oh * OUT_W
    oh = tl.where(mask, oh, 0)
    ow = tl.where(mask, ow, 0)

    ih0 = (oh * 2).to(tl.int64)
    iw0 = (ow * 2).to(tl.int64)
    base_in = n64 * stride_n + c64 * stride_c + ih0 * stride_h + iw0 * stride_w

    neg_inf = -float("inf")
    v00 = tl.load(x_ptr + base_in, mask=mask, other=neg_inf).to(tl.float32)
    v01 = tl.load(x_ptr + base_in + stride_w, mask=mask, other=neg_inf).to(tl.float32)
    v10 = tl.load(x_ptr + base_in + stride_h, mask=mask, other=neg_inf).to(tl.float32)
    v11 = tl.load(x_ptr + base_in + stride_h + stride_w, mask=mask, other=neg_inf).to(tl.float32)

    pooled = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    absx = tl.abs(pooled)
    log2e = 1.4426950408889634
    softplus = tl.maximum(pooled, 0.0) + tl.log(1.0 + tl.math.exp2(-absx * log2e))
    exp_neg2 = tl.math.exp2((-2.0 * softplus) * log2e)
    tanh_s = (1.0 - exp_neg2) / (1.0 + exp_neg2)
    out = pooled * tanh_s

    out_ptrs = (
        y_ptr
        + n64 * out_stride_n
        + c64 * out_stride_c
        + oh.to(tl.int64) * out_stride_h
        + ow.to(tl.int64) * out_stride_w
    )
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=_pointwise_xpu_autotune_configs(),
    key=["N", "C", "OUT_H", "OUT_W"],
)
@triton.jit
def _fused_pool_hardswish_mish_kernel(
    x_ptr, y_ptr,
    N, C, H, W, OUT_H, OUT_W,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    subtract_value,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_sp = tl.program_id(1)

    offs = pid_sp * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (OUT_H * OUT_W)

    n = pid_nc // C
    c = pid_nc % C
    n64 = n.to(tl.int64)
    c64 = c.to(tl.int64)

    oh = offs // OUT_W
    ow = offs - oh * OUT_W
    oh = tl.where(mask, oh, 0)
    ow = tl.where(mask, ow, 0)

    ih0 = (oh * 2).to(tl.int64)
    iw0 = (ow * 2).to(tl.int64)
    base_in = n64 * stride_n + c64 * stride_c + ih0 * stride_h + iw0 * stride_w

    neg_inf = -float("inf")
    v00 = tl.load(x_ptr + base_in, mask=mask, other=neg_inf).to(tl.float32)
    v01 = tl.load(x_ptr + base_in + stride_w, mask=mask, other=neg_inf).to(tl.float32)
    v10 = tl.load(x_ptr + base_in + stride_h, mask=mask, other=neg_inf).to(tl.float32)
    v11 = tl.load(x_ptr + base_in + stride_h + stride_w, mask=mask, other=neg_inf).to(tl.float32)

    z00 = v00 - subtract_value
    z01 = v01 - subtract_value
    z10 = v10 - subtract_value
    z11 = v11 - subtract_value

    t00 = tl.minimum(tl.maximum(z00 + 3.0, 0.0), 6.0)
    t01 = tl.minimum(tl.maximum(z01 + 3.0, 0.0), 6.0)
    t10 = tl.minimum(tl.maximum(z10 + 3.0, 0.0), 6.0)
    t11 = tl.minimum(tl.maximum(z11 + 3.0, 0.0), 6.0)

    hs00 = z00 * t00 * (1.0 / 6.0)
    hs01 = z01 * t01 * (1.0 / 6.0)
    hs10 = z10 * t10 * (1.0 / 6.0)
    hs11 = z11 * t11 * (1.0 / 6.0)

    pooled = tl.maximum(tl.maximum(hs00, hs01), tl.maximum(hs10, hs11))

    absx = tl.abs(pooled)
    log2e = 1.4426950408889634
    softplus = tl.maximum(pooled, 0.0) + tl.log(1.0 + tl.math.exp2(-absx * log2e))
    exp_neg2 = tl.math.exp2((-2.0 * softplus) * log2e)
    tanh_s = (1.0 - exp_neg2) / (1.0 + exp_neg2)
    out = pooled * tanh_s

    out_ptrs = (
        y_ptr
        + n64 * out_stride_n
        + c64 * out_stride_c
        + oh.to(tl.int64) * out_stride_h
        + ow.to(tl.int64) * out_stride_w
    )
    tl.store(out_ptrs, out.to(y_ptr.dtype.element_ty), mask=mask)


def _conv2d_bias_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)):
        raise TypeError("Expected x, weight, bias as torch.Tensors")
    if x.device.type != "xpu":
        raise RuntimeError("x must be on device='xpu'")
    if weight.device != x.device or bias.device != x.device:
        raise RuntimeError("weight and bias must be on the same XPU device as x")
    if x.ndim != 4 or weight.ndim != 4 or bias.ndim != 1:
        raise ValueError("Invalid tensor dimensions for conv2d")

    N, C_in, H, W = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in_w == C_in and (K_h, K_w) == (3, 3), "Conv parameters mismatch"
    assert bias.shape[0] == C_out

    H_out = H - 2
    W_out = W - 2
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    sxn, sxc, sxh, sxw = x.stride()
    sW_o, sW_i, sW_kh, sW_kw = weight.stride()
    syn, syc, syh, syw = y.stride()
    use_bf16 = x.dtype == torch.bfloat16

    grid = lambda META: (
        N * H_out,
        triton.cdiv(W_out, META["BLOCK_OW"]),
        triton.cdiv(C_out, META["BLOCK_OC"]),
    )
    _conv2d_nchw_3x3_bias_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W, C_out, H_out, W_out,
        sxn, sxc, sxh, sxw,
        sW_o, sW_i, sW_kh, sW_kw,
        syn, syc, syh, syw,
        USE_BF16=use_bf16,
        grf_mode="auto",
    )
    return y


def _sub_hardswish_triton(x: torch.Tensor, subtract_value) -> torch.Tensor:
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on device='xpu'")
    if not x.is_contiguous():
        x = x.contiguous()

    sv = float(subtract_value.item()) if isinstance(subtract_value, torch.Tensor) else float(subtract_value)
    y = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _hardswish_sub_kernel[grid](
        x, y, n_elements, sv,
        grf_mode="auto",
    )
    return y


def _maxpool2d_mish_triton(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on device='xpu'")
    if x.ndim != 4:
        raise ValueError("Input must be NCHW")

    N, C, H, W = x.shape
    OUT_H, OUT_W = H // 2, W // 2
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda META: (N * C, triton.cdiv(OUT_H * OUT_W, META["BLOCK_SIZE"]))
    _maxpool2d_mish_kernel[grid](
        x, y,
        N, C, H, W, OUT_H, OUT_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        grf_mode="auto",
    )
    return y


def _vendor_conv2d_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, weight, bias, stride=1, padding=0)


def _fused_pool_hardswish_mish_triton(x: torch.Tensor, subtract_value) -> torch.Tensor:
    if x.device.type != "xpu":
        raise RuntimeError("Input must be on device='xpu'")
    if x.ndim != 4:
        raise ValueError("Input must be NCHW")

    N, C, H, W = x.shape
    OUT_H, OUT_W = H // 2, W // 2
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    sv = float(subtract_value.item()) if isinstance(subtract_value, torch.Tensor) else float(subtract_value)

    grid = lambda META: (N * C, triton.cdiv(OUT_H * OUT_W, META["BLOCK_SIZE"]))
    _fused_pool_hardswish_mish_kernel[grid](
        x, y,
        N, C, H, W, OUT_H, OUT_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        sv,
        grf_mode="auto",
    )
    return y


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, subtract_value) -> torch.Tensor:
    if x.device.type != "xpu" or x.dtype != torch.float16:
        x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()

    if weight.device.type != "xpu" or weight.dtype != torch.float16:
        weight_xpu = weight.to("xpu", dtype=torch.float16).contiguous()
    else:
        weight_xpu = weight.contiguous()

    if bias.device.type != "xpu" or bias.dtype != torch.float16:
        bias_xpu = bias.to("xpu", dtype=torch.float16).contiguous()
    else:
        bias_xpu = bias.contiguous()

    y1 = _vendor_conv2d_bias(x_xpu, weight_xpu, bias_xpu)
    y3 = _fused_pool_hardswish_mish_triton(y1, subtract_value)
    return y3


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self._cached_weight_xpu = None
        self._cached_bias_xpu = None

    def _ensure_xpu_params(self):
        weight = self.conv.weight
        if (
            self._cached_weight_xpu is None
            or self._cached_weight_xpu.device.type != "xpu"
            or self._cached_weight_xpu.dtype != torch.float16
            or self._cached_weight_xpu.shape != weight.shape
        ):
            self._cached_weight_xpu = weight.detach().to("xpu", dtype=torch.float16).contiguous()

        if self.conv.bias is not None:
            bias = self.conv.bias
            if (
                self._cached_bias_xpu is None
                or self._cached_bias_xpu.device.type != "xpu"
                or self._cached_bias_xpu.dtype != torch.float16
                or self._cached_bias_xpu.shape != bias.shape
            ):
                self._cached_bias_xpu = bias.detach().to("xpu", dtype=torch.float16).contiguous()
        else:
            self._cached_bias_xpu = None

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        self._ensure_xpu_params()
        return kernel_function(x, self._cached_weight_xpu, self._cached_bias_xpu, self.subtract_value)
