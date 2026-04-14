# ruff: noqa: E731
import sys
import torch
import torch.nn as nn
import triton
import triton.language as tl


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]


def _conv3x3_xpu_autotune_configs():
    return [
        # known-good baseline family
        triton.Config({'BLOCK_CO': 16, 'BH': 8, 'BW': 8, 'GROUP_SIZE_SP': 1}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_CO': 16, 'BH': 8, 'BW': 16, 'GROUP_SIZE_SP': 1}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_CO': 16, 'BH': 16, 'BW': 8, 'GROUP_SIZE_SP': 1}, num_warps=4, num_stages=1),

        triton.Config({'BLOCK_CO': 32, 'BH': 8, 'BW': 8, 'GROUP_SIZE_SP': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_CO': 32, 'BH': 8, 'BW': 16, 'GROUP_SIZE_SP': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_CO': 32, 'BH': 16, 'BW': 8, 'GROUP_SIZE_SP': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_CO': 32, 'BH': 16, 'BW': 16, 'GROUP_SIZE_SP': 1}, num_warps=8, num_stages=1),

        triton.Config({'BLOCK_CO': 32, 'BH': 8, 'BW': 8, 'GROUP_SIZE_SP': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_CO': 32, 'BH': 8, 'BW': 16, 'GROUP_SIZE_SP': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_CO': 32, 'BH': 16, 'BW': 8, 'GROUP_SIZE_SP': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_CO': 32, 'BH': 8, 'BW': 8, 'GROUP_SIZE_SP': 4}, num_warps=8, num_stages=1),

        triton.Config({'BLOCK_CO': 64, 'BH': 8, 'BW': 8, 'GROUP_SIZE_SP': 1}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_CO': 64, 'BH': 8, 'BW': 16, 'GROUP_SIZE_SP': 1}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_CO': 64, 'BH': 16, 'BW': 8, 'GROUP_SIZE_SP': 1}, num_warps=16, num_stages=1),

        # required 32-warp / large-tile XPU configs
        triton.Config({'BLOCK_CO': 64, 'BH': 16, 'BW': 16, 'GROUP_SIZE_SP': 1}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_CO': 64, 'BH': 16, 'BW': 16, 'GROUP_SIZE_SP': 2}, num_warps=32, num_stages=1),
    ]


def _lse_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 512}, num_warps=16, num_stages=2),
        # required 32-warp large-tile config
        triton.Config({'BLOCK_M': 256}, num_warps=32, num_stages=1),
    ]


def _fused_lse_autotune_configs():
    return [
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 512}, num_warps=16, num_stages=2),
        # required 32-warp large-tile config
        triton.Config({'BLOCK_HW': 256}, num_warps=32, num_stages=1),
    ]


@triton.autotune(
    configs=_conv3x3_xpu_autotune_configs(),
    key=['Cin', 'Cout', 'H_out', 'W_out'],
)
@triton.jit
def conv3x3_nchw_fwd_kernel(x_ptr, w_ptr, b_ptr, o_ptr,
                             N, Cin, H, W, Cout, H_out, W_out,
                             stride_inN, stride_inC, stride_inH, stride_inW,
                             stride_wCout, stride_wCin, stride_wKh, stride_wKw,
                             stride_outN, stride_outC, stride_outH, stride_outW,
                             BLOCK_CO: tl.constexpr,
                             BH: tl.constexpr,
                             BW: tl.constexpr,
                             GROUP_SIZE_SP: tl.constexpr,
                             CIN: tl.constexpr,
                             KH: tl.constexpr,
                             KW: tl.constexpr,
                             grf_mode: tl.constexpr = "auto"):
    pid_n = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    pid_sp = tl.program_id(axis=2)

    num_pid_h = tl.cdiv(H_out, BH)
    num_pid_w = tl.cdiv(W_out, BW)
    num_pid_sp = num_pid_h * num_pid_w

    group_start = (pid_sp // GROUP_SIZE_SP) * GROUP_SIZE_SP
    group_size = tl.minimum(GROUP_SIZE_SP, num_pid_sp - group_start)
    pid_sp_in_group = pid_sp - group_start
    pid_sp_swizzled = group_start + (pid_sp_in_group % group_size)

    tile_h = pid_sp_swizzled // num_pid_w
    tile_w = pid_sp_swizzled % num_pid_w

    co_start = pid_co * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    offs_ho = tile_h * BH + tl.arange(0, BH)
    offs_wo = tile_w * BW + tl.arange(0, BW)

    mask_co = offs_co < Cout
    mask_hw = (offs_ho[:, None] < H_out) & (offs_wo[None, :] < W_out)

    acc = tl.zeros((BLOCK_CO, BH, BW), dtype=tl.float32)
    base_x_n = x_ptr + pid_n * stride_inN

    for ci in range(0, CIN):
        x_ci_base = base_x_n + ci * stride_inC
        for ky in range(0, KH):
            in_ho = offs_ho + ky
            x_h_base = x_ci_base + in_ho[:, None] * stride_inH
            for kx in range(0, KW):
                w_ptrs = w_ptr + (offs_co * stride_wCout + ci * stride_wCin + ky * stride_wKh + kx * stride_wKw)
                w = tl.load(w_ptrs, mask=mask_co, other=0.0).to(tl.float32)

                in_wo = offs_wo + kx
                x_ptrs = x_h_base + in_wo[None, :] * stride_inW
                x_vals = tl.load(x_ptrs, mask=mask_hw, other=0.0).to(tl.float32)
                acc += w[:, None, None] * x_vals[None, :, :]

    b = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc += b[:, None, None]

    o_ptrs = (o_ptr
              + pid_n * stride_outN
              + offs_co[:, None, None] * stride_outC
              + offs_ho[None, :, None] * stride_outH
              + offs_wo[None, None, :] * stride_outW)
    tl.store(o_ptrs, acc, mask=mask_co[:, None, None] & mask_hw[None, :, :])


@triton.jit
def groupnorm_stats_kernel(x_ptr, mean_ptr, var_ptr,
                           N, Cout, H, W,
                           G: tl.constexpr, C_PER_G: tl.constexpr, BLOCK_S: tl.constexpr):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    S = H * W
    co_start = g * C_PER_G
    base_ng = x_ptr + n * (Cout * H * W) + co_start * (H * W)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for ci in range(0, C_PER_G):
        ch_base = base_ng + ci * S
        for s in range(0, S, BLOCK_S):
            offs_s = s + tl.arange(0, BLOCK_S)
            mask_s = offs_s < S
            vals = tl.load(ch_base + offs_s, mask=mask_s, other=0.0).to(tl.float32)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    denom = C_PER_G * S
    denom_f32 = tl.full((), denom, dtype=tl.float32)
    mean = sum_val / denom_f32
    var = sum_sq / denom_f32 - mean * mean

    tl.store(mean_ptr + n * G + g, mean)
    tl.store(var_ptr + n * G + g, var)


@triton.jit
def gn_tanh_hswish_add_kernel(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, y_ptr,
                              N, Cout, H, W, eps,
                              G: tl.constexpr, C_PER_G: tl.constexpr,
                              BLOCK_CO: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    pid_sp = tl.program_id(axis=2)

    W_TILE = tl.cdiv(W, BW)
    tile_h = pid_sp // W_TILE
    tile_w = pid_sp % W_TILE

    co_start = pid_co * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    offs_ho = tile_h * BH + tl.arange(0, BH)
    offs_wo = tile_w * BW + tl.arange(0, BW)

    mask_co = offs_co < Cout
    mask_hw = (offs_ho[:, None] < H) & (offs_wo[None, :] < W)

    x_ptrs = (x_ptr
              + pid_n * (Cout * H * W)
              + offs_co[:, None, None] * (H * W)
              + offs_ho[None, :, None] * W
              + offs_wo[None, None, :])
    x = tl.load(x_ptrs, mask=mask_co[:, None, None] & mask_hw[None, :, :], other=0.0).to(tl.float32)

    g_idx = offs_co // C_PER_G
    mean_vec = tl.load(mean_ptr + pid_n * G + g_idx, mask=mask_co, other=0.0).to(tl.float32)
    var_vec = tl.load(var_ptr + pid_n * G + g_idx, mask=mask_co, other=0.0).to(tl.float32)
    inv_std_vec = 1.0 / tl.sqrt(var_vec + eps)

    gamma = tl.load(gamma_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)

    norm = (x - mean_vec[:, None, None]) * inv_std_vec[:, None, None]
    affine = norm * gamma[:, None, None] + beta[:, None, None]

    t = 2.0 * tl.sigmoid(2.0 * affine) - 1.0
    hp = tl.minimum(tl.maximum(t + 3.0, 0.0), 6.0)
    hs = t * (hp * (1.0 / 6.0))
    y = x + hs

    y_ptrs = (y_ptr
              + pid_n * (Cout * H * W)
              + offs_co[:, None, None] * (H * W)
              + offs_ho[None, :, None] * W
              + offs_wo[None, None, :])
    tl.store(y_ptrs, y, mask=mask_co[:, None, None] & mask_hw[None, :, :])


@triton.autotune(
    configs=_lse_autotune_configs(),
    key=['M', 'C', 'H', 'W'],
)
@triton.jit
def _logsumexp_dim1_kernel(x_ptr, y_ptr,
                           N, C, H, W,
                           stride_x_n, stride_x_c, stride_x_h, stride_x_w,
                           stride_y_n, stride_y_c, stride_y_h, stride_y_w,
                           M,
                           BLOCK_M: tl.constexpr,
                           grf_mode: tl.constexpr = "auto"):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    HW = H * W
    off_n = offs_m // HW
    rem = offs_m % HW
    off_h = rem // W
    off_w = rem % W

    base_x = off_n * stride_x_n + off_h * stride_x_h + off_w * stride_x_w

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for c in range(0, C):
        ptrs = x_ptr + base_x + c * stride_x_c
        x_val = tl.load(ptrs, mask=mask_m, other=-float('inf')).to(tl.float32)
        new_m = tl.maximum(m, x_val)
        s = s * tl.exp(m - new_m) + tl.exp(x_val - new_m)
        m = new_m

    out = tl.log(s) + m

    base_y = off_n * stride_y_n + off_h * stride_y_h + off_w * stride_y_w
    tl.store(y_ptr + base_y, out, mask=mask_m)


@triton.autotune(
    configs=_fused_lse_autotune_configs(),
    key=['HW_TOTAL', 'Cout', 'H', 'W'],
)
@triton.jit
def fused_gn_tanh_hswish_add_lse_kernel(
    x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, Cout, H, W, eps,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    HW_TOTAL,
    G: tl.constexpr, C_PER_G: tl.constexpr, BLOCK_HW: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW_TOTAL

    HW = H * W
    n = offs // HW
    rem = offs % HW
    h = rem // W
    w = rem % W

    row_max = tl.full((BLOCK_HW,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for c in range(0, Cout):
        g = c // C_PER_G
        mean = tl.load(mean_ptr + n * G + g, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + n * G + g, mask=mask, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)

        x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + h * stride_x_h + w * stride_x_w
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        affine = (x - mean) * inv_std * gamma + beta
        t = 2.0 * tl.sigmoid(2.0 * affine) - 1.0
        hp = tl.minimum(tl.maximum(t + 3.0, 0.0), 6.0)
        hs = t * (hp * (1.0 / 6.0))
        y = x + hs

        new_max = tl.maximum(row_max, y)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.exp(y - new_max)
        row_max = new_max

    out = tl.log(row_sum) + row_max
    out_ptrs = out_ptr + n * stride_out_n + h * stride_out_h + w * stride_out_w
    tl.store(out_ptrs, out, mask=mask)


def kernel_function(x, conv_weight, conv_bias, gn_weight, gn_bias, groups_val=16):
    assert isinstance(x, torch.Tensor), 'x must be a torch.Tensor'
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        raise RuntimeError("Tensors must be on Intel XPU (device='xpu')")

    x_xpu = x if (x.device.type == 'xpu' and x.dtype == torch.float16) else x.to('xpu', dtype=torch.float16)
    w_xpu = conv_weight if (conv_weight.device.type == 'xpu' and conv_weight.dtype == torch.float16) else conv_weight.to('xpu', dtype=torch.float16)
    b_xpu = conv_bias if (conv_bias.device.type == 'xpu' and conv_bias.dtype == torch.float16) else conv_bias.to('xpu', dtype=torch.float16)
    gnw_xpu = gn_weight if (gn_weight.device.type == 'xpu' and gn_weight.dtype == torch.float16) else gn_weight.to('xpu', dtype=torch.float16)
    gnb_xpu = gn_bias if (gn_bias.device.type == 'xpu' and gn_bias.dtype == torch.float16) else gn_bias.to('xpu', dtype=torch.float16)

    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()
    if not w_xpu.is_contiguous():
        w_xpu = w_xpu.contiguous()
    if not b_xpu.is_contiguous():
        b_xpu = b_xpu.contiguous()
    if not gnw_xpu.is_contiguous():
        gnw_xpu = gnw_xpu.contiguous()
    if not gnb_xpu.is_contiguous():
        gnb_xpu = gnb_xpu.contiguous()

    N, Cin, H, W = x_xpu.shape
    Cout, Cin_w, Kh, Kw = w_xpu.shape
    assert Cin == Cin_w and Kh == 3 and Kw == 3
    assert b_xpu.shape == (Cout,)
    assert gnw_xpu.shape == (Cout,) and gnb_xpu.shape == (Cout,)

    G = groups_val
    assert Cout % G == 0
    C_PER_G = Cout // G
    eps = 1e-5
    H_out = H - Kh + 1
    W_out = W - Kw + 1

    device = x_xpu.device
    conv_out = torch.empty((N, Cout, H_out, W_out), dtype=torch.float16, device=device)
    mean = torch.empty((N, G), dtype=torch.float32, device=device)
    var = torch.empty((N, G), dtype=torch.float32, device=device)
    y_final = torch.empty((N, 1, H_out, W_out), dtype=torch.float16, device=device)

    si_n, si_c, si_h, si_w = x_xpu.stride()
    sw_o_n, sw_o_c, sw_o_h, sw_o_w = conv_out.stride()
    sw_w_co, sw_w_ci, sw_w_kh, sw_w_kw = w_xpu.stride()

    grid_conv = lambda meta: (
        N,
        triton.cdiv(Cout, meta['BLOCK_CO']),
        triton.cdiv(H_out, meta['BH']) * triton.cdiv(W_out, meta['BW']),
    )
    conv3x3_nchw_fwd_kernel[grid_conv](
        x_xpu, w_xpu, b_xpu, conv_out,
        N, Cin, H, W, Cout, H_out, W_out,
        si_n, si_c, si_h, si_w,
        sw_w_co, sw_w_ci, sw_w_kh, sw_w_kw,
        sw_o_n, sw_o_c, sw_o_h, sw_o_w,
        CIN=Cin, KH=Kh, KW=Kw,
    )

    grid_stats = (N * G,)
    groupnorm_stats_kernel[grid_stats](
        conv_out, mean, var,
        N, Cout, H_out, W_out,
        G=G, C_PER_G=C_PER_G, BLOCK_S=1024,
        num_warps=4, num_stages=1
    )

    sy_n, sy_c, sy_h, sy_w = y_final.stride()
    hw_total = N * H_out * W_out

    def grid_fused(meta):
        return (triton.cdiv(hw_total, meta['BLOCK_HW']),)

    fused_gn_tanh_hswish_add_lse_kernel[grid_fused](
        conv_out, mean, var, gnw_xpu, gnb_xpu, y_final,
        N, Cout, H_out, W_out, eps,
        sw_o_n, sw_o_c, sw_o_h, sw_o_w,
        sy_n, sy_c, sy_h, sy_w,
        hw_total,
        G=G, C_PER_G=C_PER_G,
    )
    return y_final


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self._packed_ready = False

    def _ensure_xpu_params(self):
        if self.conv.weight.device.type != 'xpu' or self.conv.weight.dtype != torch.float16:
            self.conv.weight.data = self.conv.weight.data.to('xpu', dtype=torch.float16).contiguous()
        if self.conv.bias is not None and (self.conv.bias.device.type != 'xpu' or self.conv.bias.dtype != torch.float16):
            self.conv.bias.data = self.conv.bias.data.to('xpu', dtype=torch.float16).contiguous()
        if self.group_norm.weight.device.type != 'xpu' or self.group_norm.weight.dtype != torch.float16:
            self.group_norm.weight.data = self.group_norm.weight.data.to('xpu', dtype=torch.float16).contiguous()
        if self.group_norm.bias.device.type != 'xpu' or self.group_norm.bias.dtype != torch.float16:
            self.group_norm.bias.data = self.group_norm.bias.data.to('xpu', dtype=torch.float16).contiguous()
        self._packed_ready = True

    def forward(self, x):
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError('Intel XPU is required')
        if x.device.type != 'xpu' or x.dtype != torch.float16:
            x = x.to('xpu', dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()
        if not self._packed_ready:
            self._ensure_xpu_params()
        return kernel_function(
            x,
            self.conv.weight,
            self.conv.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
        )


def run_test():
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print('XPU device not available, skipping test.')
        sys.exit(0)

    in_ch, out_ch, k, grp = get_init_inputs()
    model = Model(in_ch, out_ch, k, grp).eval()
    x = get_inputs()[0]

    with torch.no_grad():
        x_xpu = x.to('xpu', dtype=torch.float16)
        model._ensure_xpu_params()
        x_conv = model.conv(x_xpu)
        x_norm = model.group_norm(x_conv)
        x_tanh = torch.tanh(x_norm)
        x_hs = torch.nn.functional.hardswish(x_tanh)
        ref = torch.logsumexp(x_conv + x_hs, dim=1, keepdim=True)
        out = kernel_function(
            x_xpu,
            model.conv.weight,
            model.conv.bias,
            model.group_norm.weight,
            model.group_norm.bias,
            grp,
        )
        torch.xpu.synchronize()

    if torch.allclose(out, ref, atol=2e-2, rtol=2e-2):
        print('PASS')
        sys.exit(0)
    else:
        print('FAIL')
        print('Max abs diff:', (out - ref).abs().max().item())
        sys.exit(1)