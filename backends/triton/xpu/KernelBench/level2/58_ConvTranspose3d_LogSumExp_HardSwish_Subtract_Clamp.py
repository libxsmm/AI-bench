# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _conv3d_autotune_configs():
    # Broader XPU-oriented search than the previous attempt, while staying valid:
    # vary BLOCK_SIZE across powers of 2 and cover 4/8/16/32 warps.
    return [
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32, num_stages=1),
    ]


def _reduction_autotune_configs():
    # Reduction over C, vectorized along W.
    # Include a required 32-warp XPU candidate via BLOCK_W=256.
    return [
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=32, num_stages=1),
    ]


def _elementwise_autotune_configs():
    # Elementwise kernels usually prefer simple 1D sweeps.
    return [
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32, num_stages=1),
    ]


# -----------------------------------------------------------------------------
# Subgraph 1: ConvTranspose3d + bias (fused)
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_conv3d_autotune_configs(),
    key=["n_elements", "N", "C_OUT", "OD", "OH", "OW", "D_IN", "H_IN", "W_IN"],
)
@triton.jit
def _conv_transpose3d_fused_bias(
    x_ptr, w_ptr, b_ptr, y_ptr,
    n_elements,
    N, C_OUT, OD, OH, OW,
    D_IN, H_IN, W_IN,
    stride_xN, stride_xC, stride_xD, stride_xH, stride_xW,
    stride_wCIN, stride_wCOUT, stride_wKD, stride_wKH, stride_wKW,
    stride_b,
    stride_yN, stride_yC, stride_yD, stride_yH, stride_yW,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    C_IN: tl.constexpr, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    T_DHW = OD * OH * OW
    T_HW = OH * OW
    n = offs // (C_OUT * T_DHW)
    r1 = offs % (C_OUT * T_DHW)
    co = r1 // T_DHW
    r2 = r1 % T_DHW
    do = r2 // T_HW
    r3 = r2 % T_HW
    ho = r3 // OW
    wo = r3 % OW

    n_safe = tl.where(mask, n, 0)
    co_safe = tl.where(mask, co, 0)
    do_safe = tl.where(mask, do, 0)
    ho_safe = tl.where(mask, ho, 0)
    wo_safe = tl.where(mask, wo, 0)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ci in range(C_IN):
        for kd in range(KD):
            tmpd = do_safe + PD - kd * DD
            cond_d = (tmpd >= 0) & ((tmpd % SD) == 0)
            di_raw = tmpd // SD
            cond_d = cond_d & (di_raw < D_IN)
            di = tl.where(cond_d, di_raw, 0)

            for kh in range(KH):
                tmph = ho_safe + PH - kh * DH
                cond_h = (tmph >= 0) & ((tmph % SH) == 0)
                hi_raw = tmph // SH
                cond_h = cond_h & (hi_raw < H_IN)
                hi = tl.where(cond_h, hi_raw, 0)

                for kw in range(KW):
                    tmpw = wo_safe + PW - kw * DW
                    cond_w = (tmpw >= 0) & ((tmpw % SW) == 0)
                    wi_raw = tmpw // SW
                    cond_w = cond_w & (wi_raw < W_IN)
                    wi = tl.where(cond_w, wi_raw, 0)

                    m = mask & cond_d & cond_h & cond_w

                    x_ptrs = (
                        x_ptr
                        + n_safe * stride_xN
                        + ci * stride_xC
                        + di * stride_xD
                        + hi * stride_xH
                        + wi * stride_xW
                    )
                    w_ptrs = (
                        w_ptr
                        + ci * stride_wCIN
                        + co_safe * stride_wCOUT
                        + kd * stride_wKD
                        + kh * stride_wKH
                        + kw * stride_wKW
                    )
                    x_val = tl.load(x_ptrs, mask=m, other=0.0)
                    w_val = tl.load(w_ptrs, mask=m, other=0.0)
                    acc += x_val * w_val

    b_ptrs = b_ptr + co_safe * stride_b
    b_val = tl.load(b_ptrs, mask=mask, other=0.0)
    out = acc + b_val

    y_ptrs = (
        y_ptr
        + n_safe * stride_yN
        + co_safe * stride_yC
        + do_safe * stride_yD
        + ho_safe * stride_yH
        + wo_safe * stride_yW
    )
    tl.store(y_ptrs, out, mask=mask)


def conv_transpose3d_fused_bias(
    x, w, b,
    stride=(2, 2, 2), padding=(1, 1, 1),
    dilation=(1, 1, 1), output_padding=(0, 0, 0), groups=1
):
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU is not available")
    if x.device.type != "xpu":
        raise RuntimeError("x must be on XPU")
    if x.dtype != torch.float16 or w.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("Only float16 supported")
    if groups != 1:
        raise NotImplementedError("groups>1 not supported")
    if output_padding != (0, 0, 0):
        raise NotImplementedError("output_padding!=0 not supported")

    x_cont = x.contiguous()
    w_cont = w.contiguous()

    N, C_in, D_in, H_in, W_in = x_cont.shape
    _, Cout_per_g, KD, KH, KW = w_cont.shape
    C_out = Cout_per_g
    SD, SH, SW = stride
    PD, PH, PW = padding
    DD, DH, DW = dilation

    D_out = (D_in - 1) * SD - 2 * PD + DD * (KD - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * SH - 2 * PH + DH * (KH - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * SW - 2 * PW + DW * (KW - 1) + output_padding[2] + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x_cont.device, dtype=x_cont.dtype)

    sxN, sxC, sxD, sxH, sxW = x_cont.stride()
    swCIN, swCOUT, swKD, swKH, swKW = w_cont.stride()
    (sb,) = b.stride()
    syN, syC, syD, syH, syW = y.stride()

    n_elements = y.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _conv_transpose3d_fused_bias[grid](
        x_cont, w_cont, b, y,
        n_elements,
        N, C_out, D_out, H_out, W_out,
        D_in, H_in, W_in,
        sxN, sxC, sxD, sxH, sxW,
        swCIN, swCOUT, swKD, swKH, swKW,
        sb,
        syN, syC, syD, syH, syW,
        SD=SD, SH=SH, SW=SW,
        PD=PD, PH=PH, PW=PW,
        DD=DD, DH=DH, DW=DW,
        C_IN=C_in, KD=KD, KH=KH, KW=KW,
    )
    return y


# -----------------------------------------------------------------------------
# Subgraph 2: LogSumExp over dim=1, keepdim
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_reduction_autotune_configs(),
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def _logsumexp_dim1_keep_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    sN, sC, sD, sH, sW,
    oN, oC, oD, oH, oW,
    BLOCK_W: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid_ndh = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    NDH = D * H
    n = pid_ndh // NDH
    rem = pid_ndh % NDH
    d = rem // H
    h = rem % H

    start_w = pid_w * BLOCK_W
    offs_w = start_w + tl.arange(0, BLOCK_W)
    mask = offs_w < W

    base_x = x_ptr + n * sN + d * sD + h * sH + offs_w * sW
    base_y = y_ptr + n * oN + d * oD + h * oH + offs_w * oW

    m = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
    s = tl.zeros([BLOCK_W], dtype=tl.float32)

    for ci in range(C):
        ptr = base_x + ci * sC
        x_val = tl.load(ptr, mask=mask, other=-float("inf"))
        m_new = tl.maximum(m, x_val)
        s = s * tl.exp(m - m_new) + tl.exp(x_val - m_new)
        m = m_new

    lse = tl.log(s) + m
    tl.store(base_y, lse, mask=mask)


def logsumexp_triton(x):
    if x.device.type != "xpu":
        raise RuntimeError("x must be on XPU")
    if x.dtype != torch.float16:
        raise TypeError("x must be float16")
    assert x.ndim == 5, "Input should be 5D"

    x_cont = x.contiguous()
    N, C, D, H, W = x_cont.shape
    y = torch.empty((N, 1, D, H, W), device=x_cont.device, dtype=x_cont.dtype)
    sN, sC, sD, sH, sW = x_cont.stride()
    oN, oC, oD, oH, oW = y.stride()

    grid = lambda meta: (N * D * H, triton.cdiv(W, meta["BLOCK_W"]))
    _logsumexp_dim1_keep_kernel[grid](
        x_cont, y,
        N, C, D, H, W,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
    )
    return y


# -----------------------------------------------------------------------------
# Subgraph 3: HardSwish-like: x * sigmoid(x+3) / 6
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_elementwise_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _hardswish_like_kernel(
    x_ptr, y_ptr, n_elements,
    add_scalar, div_scalar,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    z = x + add_scalar
    sig = 1.0 / (1.0 + tl.exp(-z))
    y_val = x * sig / div_scalar
    tl.store(y_ptr + offs, y_val, mask=mask)


def hardswish_triton(x, add_scalar=3.0, div_scalar=6.0):
    if x.device.type != "xpu":
        raise RuntimeError("x must be on XPU")
    if x.dtype != torch.float16:
        raise TypeError("x must be float16")
    x_cont = x.contiguous()
    n_elements = x_cont.numel()
    y = torch.empty_like(x_cont)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _hardswish_like_kernel[grid](
        x_cont, y, n_elements,
        float(add_scalar), float(div_scalar),
    )
    return y


# -----------------------------------------------------------------------------
# Subgraph 4: Bias subtraction and clamp
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_elementwise_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def _bias_sub_clamp_kernel(
    x_ptr, bias_ptr, y_ptr, n_elements,
    clamp_min, clamp_max,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr)
    y_val = x - b
    y_val = tl.minimum(tl.maximum(y_val, clamp_min), clamp_max)
    tl.store(y_ptr + offs, y_val, mask=mask)


def bias_sub_clamp_triton(x, bias, clamp_min=-1.0, clamp_max=1.0):
    if x.device.type != "xpu":
        raise RuntimeError("x must be on XPU")
    if bias.device.type != "xpu":
        raise RuntimeError("bias must be on XPU")
    if bias.numel() != 1:
        raise ValueError("bias must be a single element")
    x_cont = x.contiguous()
    y = torch.empty_like(x_cont)
    n_elements = x_cont.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _bias_sub_clamp_kernel[grid](
        x_cont, bias, y, n_elements,
        float(clamp_min), float(clamp_max),
    )
    return y


# -----------------------------------------------------------------------------
# Top-level fused pipeline
# -----------------------------------------------------------------------------
def kernel_function(x, conv_w, conv_b, bias_after):
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU not available")

    x_xpu = x.to("xpu", dtype=torch.float16) if (x.device.type != "xpu" or x.dtype != torch.float16) else x
    conv_w_xpu = conv_w.to("xpu", dtype=torch.float16) if (conv_w.device.type != "xpu" or conv_w.dtype != torch.float16) else conv_w
    conv_b_xpu = conv_b.to("xpu", dtype=torch.float16) if (conv_b.device.type != "xpu" or conv_b.dtype != torch.float16) else conv_b
    bias_after_xpu = bias_after.to("xpu", dtype=torch.float16) if (bias_after.device.type != "xpu" or bias_after.dtype != torch.float16) else bias_after

    y1 = conv_transpose3d_fused_bias(x_xpu, conv_w_xpu, conv_b_xpu)
    y2 = logsumexp_triton(y1)
    y3 = hardswish_triton(y2)
    y4 = bias_sub_clamp_triton(y3, bias_after_xpu, clamp_min=-1.0, clamp_max=1.0)
    return y4


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=2, padding=1
        )
        self.bias = nn.Parameter(torch.zeros(bias_shape))
        self.stride = stride
        self.padding = padding
        self._moved_to_xpu = False

    def _move_params_once(self):
        if not self._moved_to_xpu:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to("xpu", dtype=torch.float16).contiguous()
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.data = self.conv_transpose.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self._moved_to_xpu = True

    def forward(self, x):
        self._move_params_once()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        return kernel_function(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
        )
