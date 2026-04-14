# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Subgraph 0: Conv3D NCDHW -> bias addition
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_W': 32, 'BLOCK_H': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_W': 16, 'BLOCK_H': 16}, num_stages=2, num_warps=4),
    ],
    key=["W_OUT", "H_OUT"],
)
@triton.jit
def _conv3d_bias_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr,
    N, C_IN, D_IN, H_IN, W_IN,
    C_OUT, D_OUT, H_OUT, W_OUT,
    stride_xN, stride_xC, stride_xD, stride_xH, stride_xW,
    stride_wCo, stride_wCi, stride_wKd, stride_wKh, stride_wKw,
    stride_oN, stride_oC, stride_oD, stride_oH, stride_oW,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """
    Direct Conv3D NCDHW with bias fusion in epilogue.

    Block pointers are used for structured HxW input/output tile accesses.
    Scalar weight loads remain manual.
    """
    pid_w = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    d_out = pid_z % D_OUT
    tmp = pid_z // D_OUT
    co = tmp % C_OUT
    n = tmp // C_OUT

    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    base_x_n = x_ptr + n * stride_xN + d_out * stride_xD
    base_w_co = w_ptr + co * stride_wCo

    for ci in range(C_IN):
        base_x_nc = base_x_n + ci * stride_xC
        base_w_coci = base_w_co + ci * stride_wCi
        for kz in range(K_D):
            base_x_ncd = base_x_nc + kz * stride_xD
            base_w_cocikd = base_w_coci + kz * stride_wKd
            for ky in range(K_H):
                # Base points to the start of the 2D plane for this (n, ci, d_out + kz)
                # Offsets are in output coordinates; ky/kx are applied via tl.advance.
                x_hw_bp = tl.make_block_ptr(
                    base=base_x_ncd,
                    shape=(H_IN, W_IN),
                    strides=(stride_xH, stride_xW),
                    offsets=(h_start, w_start),
                    block_shape=(BLOCK_H, BLOCK_W),
                    order=(1, 0),
                )
                base_w_cocikdkh = base_w_cocikd + ky * stride_wKh
                for kx in range(K_W):
                    x_tile = tl.load(
                        tl.advance(x_hw_bp, (ky, kx)),
                        boundary_check=(0, 1),
                    ).to(tl.float32)
                    w_val = tl.load(base_w_cocikdkh + kx * stride_wKw).to(tl.float32)
                    acc += x_tile * w_val

    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    out_bp = tl.make_block_ptr(
        base=o_ptr + n * stride_oN + co * stride_oC + d_out * stride_oD,
        shape=(H_OUT, W_OUT),
        strides=(stride_oH, stride_oW),
        offsets=(h_start, w_start),
        block_shape=(BLOCK_H, BLOCK_W),
        order=(1, 0),
    )
    tl.store(out_bp, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def conv3d_bias(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Conv3D + bias addition on XPU.
    Enforces contiguous inputs for better memory behavior.
    """
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor)

    if x.device.type != "xpu":
        x = x.to("xpu")
    if w.device.type != "xpu":
        w = w.to("xpu")
    if b.device.type != "xpu":
        b = b.to("xpu")

    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    assert w.device == x.device and b.device == x.device
    assert x.ndim == 5 and w.ndim == 5 and b.ndim == 1

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, Cw_in, kD, kH, kW = w.shape
    assert C_in == Cw_in
    assert b.shape[0] == C_out

    D_out = D_in - (kD - 1)
    H_out = H_in - (kH - 1)
    W_out = W_in - (kW - 1)
    assert D_out > 0 and H_out > 0 and W_out > 0

    y = torch.empty((N, C_out, D_out, H_out, W_out), dtype=torch.float16, device=x.device)

    sxN, sxC, sxD, sxH, sxW = x.stride()
    swCo, swCi, swKd, swKh, swKw = w.stride()
    soN, soC, soD, soH, soW = y.stride()

    def grid(meta):
        return (
            triton.cdiv(W_out, meta["BLOCK_W"]),
            triton.cdiv(H_out, meta["BLOCK_H"]),
            N * C_out * D_out,
        )

    _conv3d_bias_kernel[grid](
        x, w, b, y,
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        sxN, sxC, sxD, sxH, sxW,
        swCo, swCi, swKd, swKh, swKw,
        soN, soC, soD, soH, soW,
        K_D=kD, K_H=kH, K_W=kW,
    )
    return y


# -----------------------------------------------------------------------------
# Subgraph 1: Mul -> Tanh -> Mul -> Sigmoid
# -----------------------------------------------------------------------------
@triton.jit
def _fused_mul_tanh_mul_sigmoid_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    n_elements, D_HW, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    nC = offsets // D_HW
    c_idx = nC % C
    c_safe = tl.where(mask, c_idx, 0)

    s = tl.load(scale_ptr + c_safe, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + c_safe, mask=mask, other=0.0).to(tl.float32)

    tmp = x * s
    tanh_tmp = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * tmp))) - 1.0
    z = tanh_tmp * b
    y = 1.0 / (1.0 + tl.exp(-z))

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_mul_tanh_mul_sigmoid(
    x: torch.Tensor,
    scaling_factor: torch.Tensor,
    bias_param: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper for fused elementwise ops on XPU.
    y = sigmoid(tanh(x * scaling_factor) * bias_param)
    Enforces contiguous tensors for better access behavior.
    """
    if x.device.type != "xpu":
        x = x.to("xpu")
    if scaling_factor.device.type != "xpu":
        scaling_factor = scaling_factor.to("xpu")
    if bias_param.device.type != "xpu":
        bias_param = bias_param.to("xpu")

    if not x.is_contiguous():
        x = x.contiguous()
    if not scaling_factor.is_contiguous():
        scaling_factor = scaling_factor.contiguous()
    if not bias_param.is_contiguous():
        bias_param = bias_param.contiguous()

    assert x.device.type == "xpu"
    N, C, D, H, W = x.shape

    sf = scaling_factor.view(-1)
    bp = bias_param.view(-1)
    assert sf.numel() == C and bp.numel() == C
    assert sf.device == x.device and bp.device == x.device

    y = torch.empty_like(x)
    n_elements = x.numel()
    D_HW = D * H * W
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_mul_tanh_mul_sigmoid_kernel[grid](
        x, sf, bp, y,
        n_elements, D_HW, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# -----------------------------------------------------------------------------
# Combined Kernel Function
# -----------------------------------------------------------------------------
def kernel_function(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scaling_factor: torch.Tensor,
    bias_param: torch.Tensor,
) -> torch.Tensor:
    """
    End-to-end kernel: Conv3D + bias -> mul->tanh->mul->sigmoid on XPU.
    Returns XPU tensor.
    """
    if x.device.type != "xpu":
        x = x.to("xpu", dtype=torch.float16)
    elif x.dtype != torch.float16:
        x = x.to(dtype=torch.float16)

    if conv_weight.device.type != "xpu":
        conv_weight = conv_weight.to("xpu", dtype=torch.float16)
    elif conv_weight.dtype != torch.float16:
        conv_weight = conv_weight.to(dtype=torch.float16)

    if conv_bias.device.type != "xpu":
        conv_bias = conv_bias.to("xpu", dtype=torch.float16)
    elif conv_bias.dtype != torch.float16:
        conv_bias = conv_bias.to(dtype=torch.float16)

    if scaling_factor.device.type != "xpu":
        scaling_factor = scaling_factor.to("xpu", dtype=torch.float16)
    elif scaling_factor.dtype != torch.float16:
        scaling_factor = scaling_factor.to(dtype=torch.float16)

    if bias_param.device.type != "xpu":
        bias_param = bias_param.to("xpu", dtype=torch.float16)
    elif bias_param.dtype != torch.float16:
        bias_param = bias_param.to(dtype=torch.float16)

    if not x.is_contiguous():
        x = x.contiguous()
    if not conv_weight.is_contiguous():
        conv_weight = conv_weight.contiguous()
    if not conv_bias.is_contiguous():
        conv_bias = conv_bias.contiguous()
    if not scaling_factor.is_contiguous():
        scaling_factor = scaling_factor.contiguous()
    if not bias_param.is_contiguous():
        bias_param = bias_param.contiguous()

    y1 = conv3d_bias(x, conv_weight, conv_bias)
    y2 = fused_mul_tanh_mul_sigmoid(y1, scaling_factor, bias_param)
    return y2


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._xpu_prepared = False

    def _prepare_xpu_params(self):
        if not self._xpu_prepared:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self.scaling_factor.data = self.scaling_factor.data.to("xpu", dtype=torch.float16).contiguous()
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float16).contiguous()
            self._xpu_prepared = True

    def forward(self, x):
        self._prepare_xpu_params()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

        return kernel_function(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )
