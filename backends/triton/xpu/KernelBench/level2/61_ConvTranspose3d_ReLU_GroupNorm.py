# ruff: noqa: E731
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _relu_groupnorm_xpu_autotune_configs():
    configs = []

    # Reduction-style kernel over spatial dimension S.
    # Sweep BLOCK_S, warps, and stages broadly for Intel XPU.
    # Keep grf_mode out of triton.Config() per XPU Triton constraint.
    for block_s in (64, 128, 256, 512, 1024, 2048):
        for num_warps in (4, 8, 16, 32):
            for num_stages in (1, 2, 3, 4):
                if block_s == 64 and num_warps > 8:
                    continue
                if block_s == 128 and num_warps > 16:
                    continue
                configs.append(
                    triton.Config(
                        {"BLOCK_S": block_s},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


# ---------------------------------------------------------------------
# Kept for compatibility/reference, but not used in the hot path.
# ---------------------------------------------------------------------
@triton.jit
def _conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    C_IN,
    C_OUT,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    stride_xn,
    stride_xc,
    stride_xd,
    stride_xh,
    stride_xw,
    stride_wci,
    stride_wco,
    stride_wkd,
    stride_wkh,
    stride_wkw,
    stride_yn,
    stride_yc,
    stride_yd,
    stride_yh,
    stride_yw,
    BLOCK_CO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CIN: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_sp = pid_sp * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_co = offs_co < C_OUT

    s_total = D_out * H_out * W_out
    mask_sp = offs_sp < s_total
    ohw = H_out * W_out
    od = offs_sp // ohw
    rem = offs_sp - od * ohw
    oh = rem // W_out
    ow = rem - oh * W_out

    acc = tl.zeros((BLOCK_CO, BLOCK_S), dtype=tl.float32)

    base_xn = pid_n.to(tl.int64) * stride_xn
    base_yn = pid_n.to(tl.int64) * stride_yn

    for ci in range(0, CIN):
        base_xci = base_xn + ci * stride_xc
        base_wci = ci * stride_wci
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_ = od - kd
                    ih_ = oh - kh
                    iw_ = ow - kw
                    in_bounds = (
                        (id_ >= 0)
                        & (id_ < D_in)
                        & (ih_ >= 0)
                        & (ih_ < H_in)
                        & (iw_ >= 0)
                        & (iw_ < W_in)
                    )
                    in_mask = mask_sp & in_bounds

                    x_ptrs = (
                        x_ptr
                        + base_xci
                        + id_ * stride_xd
                        + ih_ * stride_xh
                        + iw_ * stride_xw
                    )
                    x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0).to(tl.float32)

                    base_w = (
                        base_wci + kd * stride_wkd + kh * stride_wkh + kw * stride_wkw
                    )
                    w_ptrs = w_ptr + base_w + offs_co * stride_wco
                    w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0).to(tl.float32)

                    acc += w_vals[:, None] * x_vals[None, :]

    y_ptrs_sp = y_ptr + base_yn + od * stride_yd + oh * stride_yh + ow * stride_yw
    y_ptrs_2d = y_ptrs_sp[None, :] + offs_co[:, None] * stride_yc
    out_mask = mask_co[:, None] & mask_sp[None, :]
    tl.store(y_ptrs_2d, acc.to(tl.float16), mask=out_mask)


# ---------------------------------------------------------------------
# Fused ReLU + GroupNorm kernel retained and used.
# Expanded XPU autotuning over BLOCK_S / warps / stages.
# grf_mode remains a compiler constexpr and is supplied at launch.
# ---------------------------------------------------------------------
@triton.autotune(
    configs=_relu_groupnorm_xpu_autotune_configs(),
    key=["S", "CPG", "NUM_GROUPS"],
)
@triton.jit
def _relu_groupnorm_fwd_kernel_tiled(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    C,
    S,
    NUM_GROUPS,
    eps,
    CPG,
    GROUP_SIZE,
    BLOCK_S: tl.constexpr,
    grf_mode: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    c0 = g * CPG
    base = (n.to(tl.int64) * C + c0) * S

    sum_f32 = tl.zeros((), dtype=tl.float32)
    sumsq_f32 = tl.zeros((), dtype=tl.float32)

    for c_rel in range(0, CPG):
        chan_base = base + c_rel * S
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask = offs_s < S
            x = tl.load(x_ptr + chan_base + offs_s, mask=mask, other=0.0).to(tl.float32)
            x = tl.maximum(x, 0.0)
            sum_f32 += tl.sum(x, axis=0)
            sumsq_f32 += tl.sum(x * x, axis=0)

    inv_count = 1.0 / GROUP_SIZE
    mean = sum_f32 * inv_count
    var = sumsq_f32 * inv_count - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    for c_rel in range(0, CPG):
        ch = c0 + c_rel
        gamma = tl.load(weight_ptr + ch).to(tl.float32)
        beta = tl.load(bias_ptr + ch).to(tl.float32)
        chan_base = base + c_rel * S
        scale = gamma * inv_std
        shift = beta - mean * scale

        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask = offs_s < S
            x = tl.load(x_ptr + chan_base + offs_s, mask=mask, other=0.0).to(tl.float32)
            x = tl.maximum(x, 0.0)
            y = x * scale + shift
            tl.store(y_ptr + chan_base + offs_s, y.to(tl.float16), mask=mask)


def kernel_function(x, conv_w, gn_weight, gn_bias, num_groups=8, eps=1e-5):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU device is not available.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    for t in (conv_w, gn_weight, gn_bias):
        if not isinstance(t, torch.Tensor):
            raise TypeError("All weights must be torch.Tensors")

    x_xpu = (
        x.to("xpu", dtype=torch.float16).contiguous()
        if (x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous())
        else x
    )
    conv_w_xpu = (
        conv_w.to("xpu", dtype=torch.float16).contiguous()
        if (
            conv_w.device.type != "xpu"
            or conv_w.dtype != torch.float16
            or not conv_w.is_contiguous()
        )
        else conv_w
    )
    gn_weight_xpu = (
        gn_weight.to("xpu", dtype=torch.float16).contiguous()
        if (
            gn_weight.device.type != "xpu"
            or gn_weight.dtype != torch.float16
            or not gn_weight.is_contiguous()
        )
        else gn_weight
    )
    gn_bias_xpu = (
        gn_bias.to("xpu", dtype=torch.float16).contiguous()
        if (
            gn_bias.device.type != "xpu"
            or gn_bias.dtype != torch.float16
            or not gn_bias.is_contiguous()
        )
        else gn_bias
    )

    if x_xpu.ndim != 5 or conv_w_xpu.ndim != 5:
        raise ValueError(
            "x must be 5D [N,Cin,D,H,W], conv_w must be 5D [Cin,Cout,kD,kH,kW]"
        )

    N, C_in, D, H, W = x_xpu.shape
    w_cin, C_out, kD, kH, kW = conv_w_xpu.shape
    if w_cin != C_in:
        raise ValueError("conv_w in-channels mismatch x")
    if gn_weight_xpu.numel() != C_out or gn_bias_xpu.numel() != C_out:
        raise ValueError("gn_weight/gn_bias length must equal C_out")
    if C_out % num_groups != 0:
        raise ValueError("C_out must be divisible by num_groups")

    y1 = F.conv_transpose3d(
        x_xpu,
        conv_w_xpu,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
    )

    if not y1.is_contiguous():
        y1 = y1.contiguous()

    y = torch.empty_like(y1)

    N2, C2, D2, H2, W2 = y1.shape
    S = D2 * H2 * W2
    CPG = C2 // num_groups
    GROUP_SIZE = CPG * S

    if S == 0:
        return y
    if S < 128:
        yr = torch.relu(y1)
        yr = F.group_norm(yr, num_groups, gn_weight_xpu, gn_bias_xpu, eps)
        return yr

    grid_gn = (N2 * num_groups,)
    _relu_groupnorm_fwd_kernel_tiled[grid_gn](
        y1,
        gn_weight_xpu,
        gn_bias_xpu,
        y,
        C2,
        S,
        num_groups,
        eps,
        CPG,
        GROUP_SIZE,
        grf_mode="auto",
    )
    return y


batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=2, padding=1
        )
        self.group_norm = nn.GroupNorm(groups, out_channels)
        self.bias = bias

        self._conv_weight_xpu = None
        self._gn_weight_xpu = None
        self._gn_bias_xpu = None
        self._conv_weight_version = None
        self._gn_weight_version = None
        self._gn_bias_version = None

    def _refresh_xpu_params(self):
        conv_ver = getattr(self.conv_transpose.weight, "_version", None)
        gnw_ver = getattr(self.group_norm.weight, "_version", None)
        gnb_ver = getattr(self.group_norm.bias, "_version", None)

        need_conv = (
            self._conv_weight_xpu is None
            or self._conv_weight_version != conv_ver
            or self._conv_weight_xpu.device.type != "xpu"
            or self._conv_weight_xpu.dtype != torch.float16
            or not self._conv_weight_xpu.is_contiguous()
        )
        need_gnw = (
            self._gn_weight_xpu is None
            or self._gn_weight_version != gnw_ver
            or self._gn_weight_xpu.device.type != "xpu"
            or self._gn_weight_xpu.dtype != torch.float16
            or not self._gn_weight_xpu.is_contiguous()
        )
        need_gnb = (
            self._gn_bias_xpu is None
            or self._gn_bias_version != gnb_ver
            or self._gn_bias_xpu.device.type != "xpu"
            or self._gn_bias_xpu.dtype != torch.float16
            or not self._gn_bias_xpu.is_contiguous()
        )

        if need_conv:
            self._conv_weight_xpu = (
                self.conv_transpose.weight.detach()
                .to("xpu", dtype=torch.float16)
                .contiguous()
            )
            self._conv_weight_version = conv_ver
        if need_gnw:
            self._gn_weight_xpu = (
                self.group_norm.weight.detach()
                .to("xpu", dtype=torch.float16)
                .contiguous()
            )
            self._gn_weight_version = gnw_ver
        if need_gnb:
            self._gn_bias_xpu = (
                self.group_norm.bias.detach()
                .to("xpu", dtype=torch.float16)
                .contiguous()
            )
            self._gn_bias_version = gnb_ver

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to("xpu", dtype=torch.float16).contiguous()

        self._refresh_xpu_params()

        return kernel_function(
            x,
            self._conv_weight_xpu,
            self._gn_weight_xpu,
            self._gn_bias_xpu,
            self.group_norm.num_groups,
        )
