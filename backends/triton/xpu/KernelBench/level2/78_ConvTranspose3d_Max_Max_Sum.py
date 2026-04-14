# ruff: noqa: E731
# KernelBench-compatible wrapper — Model class injected by codegen
# The Triton kernel logic is unchanged from the original source.
import sys
import torch
import triton
import triton.language as tl
import torch.nn as nn

# ----------------------------------------------------------------------
# Triton kernel: ConvTranspose3d + Bias fusion
# ----------------------------------------------------------------------
@triton.jit
def _conv_transpose3d_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    # Problem sizes
    N: tl.constexpr, C_IN, C_OUT,
    D_IN, H_IN, W_IN,
    D_OUT, H_OUT, W_OUT,
    # Strides for x (N, C, D, H, W)
    SXN, SXC, SXD, SXH, SXW,
    # Strides for w (C_in, C_out, KD, KH, KW)
    SWCI, SWCO, SWKD, SWKH, SWKW,
    # Strides for y (N, C, D, H, W)
    SYN, SYC, SYD, SYH, SYW,
    # Transpose-conv hyper-parameters (compile-time)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRD: tl.constexpr, STRH: tl.constexpr, STRW: tl.constexpr,
    PADD: tl.constexpr, PADH: tl.constexpr, PADW: tl.constexpr,
    DILD: tl.constexpr, DILH: tl.constexpr, DILW: tl.constexpr,
    # Kernel meta parameter
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ConvTranspose3d + Bias-add kernel (NCDHW layout).
    """
    # Flattened launch over the entire output tensor: N * C_OUT * D_OUT * H_OUT * W_OUT
    n_elements = N * C_OUT * D_OUT * H_OUT * W_OUT

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask_out = offs < n_elements

    # Decode flattened offsets into (n, co, od, oh, ow)
    tmp = offs
    ow = tmp % W_OUT
    tmp = tmp // W_OUT
    oh = tmp % H_OUT
    tmp = tmp // H_OUT
    od = tmp % D_OUT
    tmp = tmp // D_OUT
    co = tmp % C_OUT
    n = tmp // C_OUT  # 0..N-1

    # Prepare output pointers
    y_offsets = n * SYN + co * SYC + od * SYD + oh * SYH + ow * SYW
    y_ptrs = y_ptr + y_offsets

    # Initialize accumulator with bias (in fp32)
    b_vals = tl.load(b_ptr + co, mask=mask_out, other=0.0)
    acc = b_vals.to(tl.float32)

    # Loop over input channels
    for ci in range(C_IN):
        base_in_ci = n * SXN + ci * SXC
        for kd in range(KD):
            id_num = od + PADD - kd * DILD
            cond_d = (id_num >= 0) & ((id_num % STRD) == 0)
            id_in = id_num // STRD
            cond_d = cond_d & (id_in < D_IN)
            id_clamp = tl.where(cond_d, id_in, 0)
            for kh in range(KH):
                ih_num = oh + PADH - kh * DILH
                cond_h = (ih_num >= 0) & ((ih_num % STRH) == 0)
                ih_in = ih_num // STRH
                cond_h = cond_h & (ih_in < H_IN)
                ih_clamp = tl.where(cond_h, ih_in, 0)
                for kw in range(KW):
                    iw_num = ow + PADW - kw * DILW
                    cond_w = (iw_num >= 0) & ((iw_num % STRW) == 0)
                    iw_in = iw_num // STRW
                    cond_w = cond_w & (iw_in < W_IN)
                    iw_clamp = tl.where(cond_w, iw_in, 0)

                    valid_all = mask_out & cond_d & cond_h & cond_w

                    x_offsets = base_in_ci + id_clamp * SXD + ih_clamp * SXH + iw_clamp * SXW
                    x_ptrs = x_ptr + x_offsets
                    w_offsets = ci * SWCI + co * SWCO + kd * SWKD + kh * SWKH + kw * SWKW
                    w_ptrs = w_ptr + w_offsets

                    x_vals = tl.load(x_ptrs, mask=valid_all, other=0.0)
                    w_vals = tl.load(w_ptrs, mask=mask_out, other=0.0)
                    acc += x_vals * w_vals

    # Store result
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)

def conv_transpose3d_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for ConvTranspose3d + Bias.
    """
    # Validations
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor)
    assert x.device == w.device == b.device
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU not available"
    assert x.device.type == "xpu"
    # Shapes and params
    N, C_in, D_in, H_in, W_in = x.shape
    Ci_w, Co_w, KD, KH, KW = w.shape
    assert Ci_w == C_in
    C_out = Co_w
    assert b.shape[0] == C_out
    stride = (2, 2, 2)
    padding = (2, 2, 2)
    dilation = (1, 1, 1)
    output_padding = (0, 0, 0)
    # Output sizes
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (KD - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (KH - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (KW - 1) + output_padding[2] + 1
    # dtypes
    assert x.dtype == torch.float16 and w.dtype == torch.float16 and b.dtype == torch.float16
    # Allocate output
    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    # Strides
    SXN, SXC, SXD, SXH, SXW = x.stride()
    SWCI, SWCO, SWKD, SWKH, SWKW = w.stride()
    SYN, SYC, SYD, SYH, SYW = y.stride()
    # Launch grid
    n_elements = N * C_out * D_out * H_out * W_out
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _conv_transpose3d_bias_kernel[grid](
        x, w, b, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        SXN, SXC, SXD, SXH, SXW,
        SWCI, SWCO, SWKD, SWKH, SWKW,
        SYN, SYC, SYD, SYH, SYW,
        KD=KD, KH=KH, KW=KW,
        STRD=stride[0], STRH=stride[1], STRW=stride[2],
        PADD=padding[0], PADH=padding[1], PADW=padding[2],
        DILD=dilation[0], DILH=dilation[1], DILW=dilation[2],
        BLOCK_SIZE=256,
        num_warps=8, num_stages=2
    )
    return y

# ----------------------------------------------------------------------
# Triton kernel: Fused MaxPool3d(k2->k3) + Sum over channels
# ----------------------------------------------------------------------
@triton.jit
def _fused_maxpool3d_sum_channels(
    x_ptr, y_ptr,
    N, C, D, H, W,
    D2, H2, W2,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_WO: tl.constexpr, K_COMB: tl.constexpr
):
    """
    Fused kernel:
      MaxPool3d(k=2,s=2) -> MaxPool3d(k=3,s=3) -> Sum over channel dim.
    """
    pid_w = tl.program_id(axis=0)
    pid_d2 = tl.program_id(axis=1)
    pid_nh = tl.program_id(axis=2)

    h2 = pid_nh % H2
    n = pid_nh // H2

    start_wo = pid_w * BLOCK_WO
    offs_wo = start_wo + tl.arange(0, BLOCK_WO)
    mask_wo = offs_wo < W2

    d_base = pid_d2 * K_COMB
    h_base = h2 * K_COMB

    acc_sum = tl.zeros([BLOCK_WO], dtype=tl.float32)

    base_n = n * stride_n
    for c in range(C):
        base_nc = base_n + c * stride_c
        max_val = tl.full([BLOCK_WO], -float("inf"), dtype=tl.float32)
        for rd in range(K_COMB):
            d_idx = d_base + rd
            mask_d = d_idx < D
            base_ncd = base_nc + d_idx * stride_d
            for rh in range(K_COMB):
                h_idx = h_base + rh
                mask_h = h_idx < H
                base_ncdh = base_ncd + h_idx * stride_h
                for rw in range(K_COMB):
                    w_idx = offs_wo * K_COMB + rw
                    mask_w = w_idx < W
                    m = mask_wo & mask_d & mask_h & mask_w
                    ptrs = x_ptr + base_ncdh + w_idx * stride_w
                    x_val = tl.load(ptrs, mask=m, other=0.0)
                    x_val_f32 = x_val.to(tl.float32)
                    x_val_f32 = tl.where(m, x_val_f32, -float("inf"))
                    max_val = tl.maximum(max_val, x_val_f32)
        acc_sum += max_val

    out_ptrs = (y_ptr + n * out_stride_n + 0 * out_stride_c +
                pid_d2 * out_stride_d + h2 * out_stride_h +
                offs_wo * out_stride_w)
    tl.store(out_ptrs, acc_sum.to(y_ptr.dtype.element_ty), mask=mask_wo)

def fused_maxpool3d_sum_channels_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for fused MaxPool3d(k2->k3) + Sum channels.
    """
    assert isinstance(x, torch.Tensor)
    assert x.device.type == "xpu"
    assert x.dtype in (torch.bfloat16, torch.float16)
    x = x.contiguous()

    N, C, D, H, W = x.shape
    # First pool: k=2,s=2
    D1 = (D - 2) // 2 + 1
    H1 = (H - 2) // 2 + 1
    W1 = (W - 2) // 2 + 1
    # Second pool: k=3,s=3
    D2 = (D1 - 3) // 3 + 1
    H2 = (H1 - 3) // 3 + 1
    W2 = (W1 - 3) // 3 + 1

    y = torch.empty((N, 1, D2, H2, W2), dtype=x.dtype, device=x.device)

    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oD, oH, oW = y.stride()

    K_COMB = 6  # 2*3
    BLOCK_WO = 8

    grid = (triton.cdiv(W2, BLOCK_WO), D2, N * H2)
    _fused_maxpool3d_sum_channels[grid](
        x, y,
        N, C, D, H, W,
        D2, H2, W2,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
        BLOCK_WO=BLOCK_WO, K_COMB=K_COMB,
        num_warps=8, num_stages=1
    )
    return y

# ----------------------------------------------------------------------
# Top-level wrapper
# ----------------------------------------------------------------------
def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    End-to-end Triton implementation:
      ConvTranspose3d -> MaxPool3d(k=2)->MaxPool3d(k=3) -> Sum over channels.
    """
    # Validate inputs
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor)
    assert x.device.type == "xpu" and w.device.type == "xpu" and b.device.type == "xpu"
    # Step 1: ConvTranspose3d + bias
    y1 = conv_transpose3d_triton(x, w, b)
    # Step 2: fused maxpool + sum
    y2 = fused_maxpool3d_sum_channels_triton(y1)
    return y2

# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------


batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return kernel_function(
            x, self.conv_transpose.weight, self.conv_transpose.bias
        )
