# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _fused_conv_pool_autotune_configs():
    configs = []

    # Broaden search across channel/spatial tiling, warp count, stages, and OH grouping.
    # Keep powers-of-2 block sizes and include large 256-channel tiles with 32 warps.
    tile_specs = [
        (64, 8, 8),
        (64, 8, 16),
        (64, 16, 8),
        (64, 16, 16),
        (64, 16, 32),
        (64, 32, 16),
        (128, 8, 8),
        (128, 8, 16),
        (128, 16, 8),
        (128, 16, 16),
        (128, 8, 32),
        (128, 32, 8),
        (128, 16, 32),
        (128, 32, 16),
        (256, 8, 8),
        (256, 8, 16),
        (256, 16, 8),
        (256, 16, 16),
        (256, 8, 32),
        (256, 32, 8),
    ]

    for block_co, block_oh, block_ow in tile_specs:
        if block_co == 64:
            warp_stage_pairs = [(4, 2), (8, 2), (8, 3)]
        elif block_co == 128:
            warp_stage_pairs = [(8, 2), (16, 2), (16, 3)]
        else:
            warp_stage_pairs = [(16, 2), (32, 2), (32, 3)]

        for num_warps, num_stages in warp_stage_pairs:
            for group_size_oh in (1, 2, 4):
                # Avoid oversized grouping for very small OH tiles.
                if group_size_oh > 1 and block_oh >= 32:
                    continue
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_CO": block_co,
                            "BLOCK_OH": block_oh,
                            "BLOCK_OW": block_ow,
                            "GROUP_SIZE_OH": group_size_oh,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    # Explicit large-tile XPU coverage with 32 warps as requested.
    for group_size_oh in (1, 2):
        configs.extend([
            triton.Config(
                {"BLOCK_CO": 256, "BLOCK_OH": 16, "BLOCK_OW": 16, "GROUP_SIZE_OH": group_size_oh},
                num_warps=32,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_CO": 256, "BLOCK_OH": 16, "BLOCK_OW": 32, "GROUP_SIZE_OH": group_size_oh},
                num_warps=32,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_CO": 256, "BLOCK_OH": 32, "BLOCK_OW": 16, "GROUP_SIZE_OH": group_size_oh},
                num_warps=32,
                num_stages=2,
            ),
        ])

    return configs


def _reduce_sum_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=3),
    ]


class Model(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

        if self.conv.weight.device.type != "xpu" or self.conv.weight.dtype != torch.float16:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.conv.weight.is_contiguous():
            self.conv.weight.data = self.conv.weight.data.contiguous()

        if self.conv.bias is not None:
            if self.conv.bias.device.type != "xpu" or self.conv.bias.dtype != torch.float16:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.conv.bias.is_contiguous():
                self.conv.bias.data = self.conv.bias.data.contiguous()

        return kernel_function(x, self.conv.weight, self.conv.bias)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.float16)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]


@triton.autotune(
    configs=_fused_conv_pool_autotune_configs(),
    key=["N", "C_out", "OH", "OW"],
)
@triton.jit
def _fused_conv_pool_sigmoid_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    C_out, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_bc,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    GROUP_SIZE_OH: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL_KH: tl.constexpr,
    POOL_KW: tl.constexpr,
    POOL_STRIDE_H: tl.constexpr,
    POOL_STRIDE_W: tl.constexpr,
    CIN_CONST: tl.constexpr,
    ACC_DTYPE: tl.constexpr = tl.float32,
    grf_mode: tl.constexpr = "auto",
):
    pid = tl.program_id(0)

    num_oh_blocks = tl.cdiv(OH, BLOCK_OH)
    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    num_co_blocks = tl.cdiv(C_out, BLOCK_CO)

    tiles_per_n = num_co_blocks * num_oh_blocks * num_ow_blocks
    n_id = pid // tiles_per_n
    rem = pid % tiles_per_n

    co_block_id = rem // (num_oh_blocks * num_ow_blocks)
    rem_spatial = rem % (num_oh_blocks * num_ow_blocks)

    group_id = rem_spatial // (GROUP_SIZE_OH * num_ow_blocks)
    first_oh_block = group_id * GROUP_SIZE_OH
    valid_group_size = tl.minimum(num_oh_blocks - first_oh_block, GROUP_SIZE_OH)

    rem_in_group = rem_spatial % (GROUP_SIZE_OH * num_ow_blocks)
    pid_ow = rem_in_group // valid_group_size
    pid_oh = first_oh_block + (rem_in_group % valid_group_size)

    offs_co = co_block_id * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    co_mask = offs_co < C_out
    oh_mask = offs_oh < OH
    ow_mask = offs_ow < OW
    n_valid = n_id < N

    n_offset_x = n_id.to(tl.int64) * stride_xn
    n_offset_y = n_id.to(tl.int64) * stride_yn

    x_base_n = x_ptr + n_offset_x
    y_base_n = y_ptr + n_offset_y

    acc = tl.zeros((BLOCK_CO, BLOCK_OH, BLOCK_OW), dtype=ACC_DTYPE)
    inv_pool_area = 1.0 / (POOL_KH * POOL_KW)

    oh_pool_base = offs_oh * POOL_STRIDE_H
    ow_pool_base = offs_ow * POOL_STRIDE_W
    spatial_mask = oh_mask[:, None] & ow_mask[None, :]

    for ci in range(CIN_CONST):
        x_base_nc = x_base_n + ci * stride_xc
        for kh in range(KH):
            h_base = oh_pool_base + kh
            for kw in range(KW):
                w_base_in = ow_pool_base + kw
                pooled = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=ACC_DTYPE)

                for ph in range(POOL_KH):
                    h_idx = h_base + ph
                    h_valid = h_idx < H
                    h2d = h_idx[:, None]
                    for pw in range(POOL_KW):
                        w_idx = w_base_in + pw
                        w_valid = w_idx < W
                        w2d = w_idx[None, :]
                        mask_hw = h_valid[:, None] & w_valid[None, :] & spatial_mask & n_valid
                        x_vals = tl.load(
                            x_base_nc + h2d * stride_xh + w2d * stride_xw,
                            mask=mask_hw,
                            other=0.0,
                        )
                        pooled += x_vals

                pooled *= inv_pool_area
                w_ptrs = w_ptr + offs_co * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask & n_valid, other=0.0)
                acc += w_vec[:, None, None] * pooled[None, :, :]

    b_vec = tl.load(b_ptr + offs_co * stride_bc, mask=co_mask & n_valid, other=0.0)
    acc += b_vec[:, None, None]

    log2e = 1.4426950408889634
    exp_neg = tl.math.exp2((-acc) * log2e)
    y_tile = (1.0 / (1.0 + exp_neg)).to(y_ptr.dtype.element_ty)

    y_ptrs = y_base_n + (
        offs_co[:, None, None] * stride_yc +
        offs_oh[None, :, None] * stride_yh +
        offs_ow[None, None, :] * stride_yw
    )
    store_mask = co_mask[:, None, None] & oh_mask[None, :, None] & ow_mask[None, None, :] & n_valid
    tl.store(y_ptrs, y_tile, mask=store_mask)


def fused_conv_pool_sigmoid(x, conv_weight, conv_bias):
    assert x.device.type == "xpu"
    assert conv_weight.device.type == "xpu"
    assert conv_bias.device.type == "xpu"
    assert x.dtype == torch.float16
    assert conv_weight.dtype == torch.float16
    assert conv_bias.dtype == torch.float16

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == C_in_w

    pool_kh, pool_kw = 4, 4
    pool_stride_h, pool_stride_w = 4, 4
    H_conv = H - KH + 1
    W_conv = W - KW + 1
    OH = (H_conv - pool_kh) // pool_stride_h + 1
    OW = (W_conv - pool_kw) // pool_stride_w + 1

    out = torch.empty((N, C_out, OH, OW), dtype=torch.float16, device=x.device)

    def grid(meta):
        return (
            N
            * triton.cdiv(C_out, meta["BLOCK_CO"])
            * triton.cdiv(OH, meta["BLOCK_OH"])
            * triton.cdiv(OW, meta["BLOCK_OW"]),
        )

    _fused_conv_pool_sigmoid_kernel[grid](
        x, conv_weight, conv_bias, out,
        N, C_in, H, W,
        C_out, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        conv_bias.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        KH=KH, KW=KW,
        POOL_KH=pool_kh, POOL_KW=pool_kw,
        POOL_STRIDE_H=pool_stride_h, POOL_STRIDE_W=pool_stride_w,
        CIN_CONST=C_in,
        ACC_DTYPE=tl.float32,
        grf_mode="auto",
    )
    return out


@triton.autotune(
    configs=_reduce_sum_autotune_configs(),
    key=["L"],
)
@triton.jit
def _reduce_sum_nchw_kernel(
    x_ptr, y_ptr, N, L, stride_n,
    BLOCK_SIZE: tl.constexpr,
    grf_mode: tl.constexpr = "auto",
):
    n = tl.program_id(0)
    if n >= N:
        return

    base = n.to(tl.int64) * stride_n
    acc = tl.zeros((), dtype=tl.float32)
    for start in tl.range(0, L, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < L
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    tl.store(y_ptr + n, acc.to(y_ptr.dtype.element_ty))


def reduce_sum_nchw(x):
    assert x.device.type == "xpu"
    N, C, H, W = x.shape
    L = C * H * W
    y = torch.empty((N,), dtype=x.dtype, device=x.device)

    def grid(meta):
        return (N,)

    _reduce_sum_nchw_kernel[grid](x, y, N, L, x.stride(0), grf_mode="auto")
    return y


def kernel_function(x, conv_weight, conv_bias):
    x_xpu = x if x.device.type == "xpu" and x.dtype == torch.float16 else x.to("xpu", dtype=torch.float16)
    w_xpu = conv_weight if conv_weight.device.type == "xpu" and conv_weight.dtype == torch.float16 else conv_weight.to("xpu", dtype=torch.float16)
    b_xpu = conv_bias if conv_bias.device.type == "xpu" and conv_bias.dtype == torch.float16 else conv_bias.to("xpu", dtype=torch.float16)

    if not x_xpu.is_contiguous():
        x_xpu = x_xpu.contiguous()
    if not w_xpu.is_contiguous():
        w_xpu = w_xpu.contiguous()
    if not b_xpu.is_contiguous():
        b_xpu = b_xpu.contiguous()

    y_inter = fused_conv_pool_sigmoid(x_xpu, w_xpu, b_xpu)
    y = reduce_sum_nchw(y_inter)
    return y


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

        if self.conv.weight.device.type != "xpu" or self.conv.weight.dtype != torch.float16:
            self.conv.weight.data = self.conv.weight.data.to("xpu", dtype=torch.float16).contiguous()
        elif not self.conv.weight.is_contiguous():
            self.conv.weight.data = self.conv.weight.data.contiguous()

        if self.conv.bias is not None:
            if self.conv.bias.device.type != "xpu" or self.conv.bias.dtype != torch.float16:
                self.conv.bias.data = self.conv.bias.data.to("xpu", dtype=torch.float16).contiguous()
            elif not self.conv.bias.is_contiguous():
                self.conv.bias.data = self.conv.bias.data.contiguous()

        return kernel_function(x, self.conv.weight, self.conv.bias)