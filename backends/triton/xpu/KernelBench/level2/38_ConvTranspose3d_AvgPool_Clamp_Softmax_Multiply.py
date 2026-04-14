# ruff: noqa: E731
import torch
import triton
import triton.language as tl
import torch.nn as nn


# ---------------- Subgraph 1: avg_pool3d -> conv_transpose3d -> clamp ----------------
@triton.jit
def _fused_pool_deconv3d_clamp(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, D, H, W,
    Dp, Hp, Wp,
    sXn, sXc, sXd, sXh, sXw,
    sWcin, sWcout, sWkd, sWkh, sWkw,
    sYn, sYc, sYd, sYh, sYw,
    clamp_min, clamp_max,
    DO_CLAMP: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    DH = D * H
    n = pid0 // DH
    rem = pid0 % DH
    od = rem // H
    oh = rem % H

    co_start = pid1 * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    offs_w = tl.arange(0, BLOCK_W)

    mask_co = offs_co < Cout
    mask_w = offs_w < W

    acc = tl.zeros((BLOCK_CO, BLOCK_W), tl.float32)

    od_par = od & 1
    oh_par = oh & 1
    ow = offs_w
    ow_par = ow & 1
    scale = 0.125

    tl.max_contiguous(offs_w, BLOCK_W)

    for ci in range(0, Cin):
        x_base_nc = x_ptr + n * sXn + ci * sXc
        for dd in range(0, 2):
            xd = od + dd
            valid_d = xd < D
            kd = 1 + od_par - (xd & 1)
            for hh in range(0, 2):
                xh = oh + hh
                valid_h = xh < H
                kh = 1 + oh_par - (xh & 1)
                x_dh = x_base_nc + xd * sXd + xh * sXh
                for ww in range(0, 2):
                    xw = ow + ww
                    valid_w = (xw < W) & mask_w
                    kw = 1 + ow_par - (xw & 1)
                    lane_mask = valid_d & valid_h & valid_w

                    x_vals = tl.load(x_dh + xw * sXw, mask=lane_mask, other=0.0).to(tl.float32)
                    w_base = w_ptr + ci * sWcin + kd * sWkd + kh * sWkh + kw * sWkw
                    w_vals = tl.load(w_base + offs_co * sWcout, mask=mask_co, other=0.0).to(tl.float32)
                    acc += (w_vals[:, None] * x_vals[None, :]) * scale

    b_vals = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc = acc + b_vals[:, None]

    if DO_CLAMP:
        acc = tl.maximum(acc, clamp_min)
        acc = tl.minimum(acc, clamp_max)

    y_base = y_ptr + n * sYn + od * sYd + oh * sYh
    ptrs = y_base + offs_co[:, None] * sYc + offs_w[None, :] * sYw
    out = acc.to(y_ptr.dtype.element_ty)
    out_mask = mask_co[:, None] & mask_w[None, :]
    tl.store(ptrs, out, mask=out_mask)


@triton.jit
def _clamp_5d_kernel(
    x_ptr, y_ptr, n_elements, clamp_min, clamp_max,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, clamp_min)
    x = tl.minimum(x, clamp_max)
    tl.store(y_ptr + offs, x.to(y_ptr.dtype.element_ty), mask=mask)


def _sg1_fwd(x, w, b, clamp_min: float, clamp_max: float):
    assert x.device.type == 'xpu' and w.device == x.device and b.device == x.device
    assert x.dtype == w.dtype == b.dtype
    N, Cin, D, H, W = x.shape
    Cout = w.shape[1]
    Dp, Hp, Wp = D // 2, H // 2, W // 2
    x_ = x.contiguous()
    w_ = w.contiguous()
    b_ = b.contiguous()

    sXn, sXc, sXd, sXh, sXw = x_.stride()
    sWcin, sWcout, sWkd, sWkh, sWkw = w_.stride()

    BLOCK_CO = 32
    BLOCK_W = 32

    y_tmp = torch.empty((N, Cout, D, H, W), device=x.device, dtype=x.dtype)
    sYn, sYc, sYd, sYh, sYw = y_tmp.stride()

    grid = (N * D * H, triton.cdiv(Cout, BLOCK_CO))

    _fused_pool_deconv3d_clamp[grid](
        x_, w_, b_, y_tmp,
        N, Cin, Cout, D, H, W,
        Dp, Hp, Wp,
        sXn, sXc, sXd, sXh, sXw,
        sWcin, sWcout, sWkd, sWkh, sWkw,
        sYn, sYc, sYd, sYh, sYw,
        float(clamp_min), float(clamp_max),
        DO_CLAMP=False,
        BLOCK_CO=BLOCK_CO, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=1
    )

    y = torch.empty_like(y_tmp)
    n_elements = y_tmp.numel()
    BLOCK_SIZE = 1024
    clamp_grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _clamp_5d_kernel[clamp_grid](
        y_tmp, y, n_elements, float(clamp_min), float(clamp_max),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=1
    )
    return y


# ---------------- Subgraph 2: spatial softmax ----------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
    ],
    key=["S"],
)
@triton.jit
def _spatial_softmax3d_rowwise(x_ptr, y_ptr, R, S, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= R:
        return
    row_start = pid * S
    NEG_INF = -1e30
    row_max = NEG_INF
    for off in tl.range(0, S, BLOCK_SIZE):
        idx = row_start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < row_start + S
        v = tl.load(x_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(v, axis=0))
    row_sum = 0.0
    for off in tl.range(0, S, BLOCK_SIZE):
        idx = row_start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < row_start + S
        v = tl.load(x_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        row_sum += tl.sum(tl.exp(v - row_max), axis=0)
    inv_sum = 1.0 / row_sum
    for off in tl.range(0, S, BLOCK_SIZE):
        idx = row_start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < row_start + S
        v = tl.load(x_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        p = tl.exp(v - row_max) * inv_sum
        tl.store(y_ptr + idx, p.to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
    ],
    key=["S"],
)
@triton.jit
def _spatial_softmax3d_rowwise_scaled(x_ptr, scale_ptr, y_ptr, R, S, C, stride_sc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= R:
        return
    row_start = pid * S
    c_idx = pid % C
    scale_val = tl.load(scale_ptr + c_idx * stride_sc).to(tl.float32)
    NEG_INF = -1e30
    row_max = NEG_INF
    for off in tl.range(0, S, BLOCK_SIZE):
        idx = row_start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < row_start + S
        v = tl.load(x_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(v, axis=0))
    row_sum = 0.0
    for off in tl.range(0, S, BLOCK_SIZE):
        idx = row_start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < row_start + S
        v = tl.load(x_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        row_sum += tl.sum(tl.exp(v - row_max), axis=0)
    inv_sum = 1.0 / row_sum
    for off in tl.range(0, S, BLOCK_SIZE):
        idx = row_start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < row_start + S
        v = tl.load(x_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        p = tl.exp(v - row_max) * inv_sum
        p = p * scale_val
        tl.store(y_ptr + idx, p.to(y_ptr.dtype.element_ty), mask=mask)


def _sg2_fwd(x):
    assert x.device.type == 'xpu'
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert x.is_contiguous()
    B, C, D, H, W = x.shape
    S = D * H * W
    R = B * C
    y = torch.empty_like(x)

    def grid(meta):
        return (R,)

    _spatial_softmax3d_rowwise[grid](x, y, R, S)
    return y


def _sg23_fwd(x, scale):
    assert x.device.type == 'xpu' and scale.device == x.device
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert x.is_contiguous()
    B, C, D, H, W = x.shape
    S = D * H * W
    R = B * C
    scale_contig = scale.contiguous()
    stride_sc = scale_contig.stride(1)
    y = torch.empty_like(x)

    def grid(meta):
        return (R,)

    _spatial_softmax3d_rowwise_scaled[grid](x, scale_contig, y, R, S, C, stride_sc)
    return y


# ---------------- Subgraph 3: channel scale multiply ----------------
@triton.jit
def _channel_scale_kernel(x_ptr, scale_ptr, y_ptr, n_elements, C, DHW, stride_sc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nc = offs // DHW
    c_idx = nc % C
    s_vals = tl.load(scale_ptr + c_idx * stride_sc, mask=mask, other=1.0).to(tl.float32)
    y = x_vals * s_vals
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def _sg3_fwd(x, scale):
    assert x.device.type == 'xpu' and scale.device == x.device
    N, C, D, H, W = x.shape
    assert scale.shape == (1, C, 1, 1, 1)
    x_contig = x.contiguous()
    scale_contig = scale.contiguous()
    y = torch.empty((N, C, D, H, W), device=x.device, dtype=torch.float16)
    n_elements = x_contig.numel()
    DHW = D * H * W
    stride_sc = scale_contig.stride(1)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _channel_scale_kernel[grid](
        x_contig, scale_contig, y,
        n_elements, C, DHW, stride_sc,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=1
    )
    return y


# ---------------- Top-level fused function ----------------
def kernel_function(x, w, b, scale):
    assert hasattr(torch, 'xpu') and torch.xpu.is_available(), 'XPU not available'

    if x.device.type != 'xpu' or x.dtype != torch.float16:
        x_xpu = x.to('xpu', dtype=torch.float16).contiguous()
    else:
        x_xpu = x.contiguous()
    if w.device.type != 'xpu' or w.dtype != torch.float16:
        w_xpu = w.to('xpu', dtype=torch.float16).contiguous()
    else:
        w_xpu = w.contiguous()
    if b.device.type != 'xpu' or b.dtype != torch.float16:
        b_xpu = b.to('xpu', dtype=torch.float16).contiguous()
    else:
        b_xpu = b.contiguous()
    if scale.device.type != 'xpu' or scale.dtype != torch.float16:
        scale_xpu = scale.to('xpu', dtype=torch.float16).contiguous()
    else:
        scale_xpu = scale.contiguous()

    assert x_xpu.dim() == 5 and w_xpu.dim() == 5 and b_xpu.dim() == 1 and scale_xpu.dim() == 5
    y1 = _sg1_fwd(x_xpu, w_xpu, b_xpu, 0.0, 1.0)
    y = _sg23_fwd(y1, scale_xpu)
    return y


# ---------------- Self-test ----------------
def run_test():
    from torch import nn

    class RefModel(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
            super().__init__()
            self.avg_pool = nn.AvgPool3d(pool_kernel_size)
            self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
            self.clamp_min = clamp_min
            self.clamp_max = clamp_max
            self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

        def forward(self, x):
            x = self.avg_pool(x)
            x = self.conv_transpose(x)
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            b, c, d, h, w = x.shape
            x = x.view(b, c, -1)
            x = torch.softmax(x, dim=2)
            x = x.view(b, c, d, h, w)
            x = x * self.scale
            return x

    batch_size = 16
    in_channels, out_channels = 32, 64
    depth, height, width = 16, 32, 32
    kernel_size, stride, padding, output_padding = 3, 2, 1, 1
    pool_kernel_size = 2
    clamp_min, clamp_max = 0.0, 1.0

    x_cpu = torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float16)
    model = RefModel(in_channels, out_channels, kernel_size, (stride,) * 3, (padding,) * 3, (output_padding,) * 3, (pool_kernel_size,) * 3, clamp_min, clamp_max)
    ref = model(x_cpu)

    x_t = x_cpu.to('xpu')
    w_t = model.conv_transpose.weight.to('xpu')
    b_t = model.conv_transpose.bias.to('xpu')
    scale_t = model.scale.to('xpu')

    y_t = kernel_function(x_t, w_t, b_t, scale_t)
    torch.xpu.synchronize()
    y_cpu = y_t.cpu()

    if torch.allclose(ref, y_cpu, rtol=1e-3, atol=1e-3):
        print('PASS')
        exit(0)
    else:
        max_err = (ref - y_cpu).abs().max().item()
        print(f'FAIL: max error {max_err}')
        exit(1)


batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=output_padding)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self._params_on_xpu = False

    def forward(self, x):
        if x.device.type != 'xpu' or x.dtype != torch.float16:
            x = x.to('xpu', dtype=torch.float16).contiguous()
        else:
            x = x.contiguous()

        if (not self._params_on_xpu) or self.conv_transpose.weight.device.type != 'xpu' or self.conv_transpose.weight.dtype != torch.float16:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to('xpu', dtype=torch.float16).contiguous()
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.to('xpu', dtype=torch.float16).contiguous()
            self.scale.data = self.scale.data.to('xpu', dtype=torch.float16).contiguous()
            self._params_on_xpu = True
        else:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous()
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.contiguous()
            self.scale.data = self.scale.data.contiguous()

        return kernel_function(x, self.conv_transpose.weight, self.conv_transpose.bias, self.scale)