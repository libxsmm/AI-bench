import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Spatial-tiled Conv2d + tanh + scale + bias (NHWC layout, block_ptr) ----------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W', 'C_IN', 'C_out', 'OH', 'OW'],
)
@triton.jit
def _conv2d_tanh_scale_bias_spatial(
    x_ptr, w_ptr, conv_bias_ptr, add_bias_ptr, y_ptr,
    N_batch, H, W, C_out, OH, OW,
    stride_wkh, stride_wkw, stride_wci, stride_wco,
    scaling_factor,
    BLOCK_OW: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, C_IN: tl.constexpr,
):
    n = tl.program_id(0)
    oh = tl.program_id(1)
    pid_ow_n = tl.program_id(2)
    num_ow_tiles = tl.cdiv(OW, BLOCK_OW)
    pid_ow = pid_ow_n % num_ow_tiles
    pid_n = pid_ow_n // num_ow_tiles
    ow0 = pid_ow * BLOCK_OW
    HW = H * W

    acc = tl.zeros((BLOCK_OW, BLOCK_N), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            x_row = n * HW + (oh + kh) * W + (ow0 + kw)
            x_bp = tl.make_block_ptr(
                base=x_ptr, shape=(x_row + W - (ow0 + kw), C_IN),
                strides=(C_IN, 1), offsets=(x_row, 0),
                block_shape=(BLOCK_OW, BLOCK_K), order=(1, 0),
            )
            w_bp = tl.make_block_ptr(
                base=w_ptr + kh * stride_wkh + kw * stride_wkw,
                shape=(C_IN, C_out), strides=(stride_wci, stride_wco),
                offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
            )
            for c0 in range(0, C_IN, BLOCK_K):
                x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                w_tile = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
                x_bp = tl.advance(x_bp, (0, BLOCK_K))
                w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    # conv bias
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    cb = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += cb[None, :]

    # tanh
    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0

    # scale
    acc = acc * scaling_factor

    # add per-channel bias
    ab = tl.load(add_bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += ab[None, :]

    # store
    OHOW = OH * OW
    y_row = n * OHOW + oh * OW + ow0
    y_bp = tl.make_block_ptr(
        base=y_ptr, shape=(y_row + OW - ow0, C_out),
        strides=(C_out, 1), offsets=(y_row, pid_n * BLOCK_N),
        block_shape=(BLOCK_OW, BLOCK_N), order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# ---------- Triton MaxPool2d kernel (NHWC input/output) ----------
@triton.jit
def _maxpool2d_nhwc_kernel(
    x_ptr, y_ptr,
    N_batch, OH_in, OW_in, C,
    pool_h, pool_w,
    OH_out, OW_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (N_batch, OH_out, OW_out * ceil(C/BLOCK_C))
    n = tl.program_id(0)
    oh_out = tl.program_id(1)
    pid2 = tl.program_id(2)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    ow_out = pid2 // num_c_tiles
    pid_c = pid2 % num_c_tiles

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    neg_inf = -float("inf")
    max_val = tl.full((BLOCK_C,), neg_inf, dtype=tl.float32)

    ih_start = oh_out * pool_h
    iw_start = ow_out * pool_w

    for ph in range(pool_h):
        for pw in range(pool_w):
            ih = ih_start + ph
            iw = iw_start + pw
            idx = n * OH_in * OW_in * C + ih * OW_in * C + iw * C + c_offs
            val = tl.load(x_ptr + idx, mask=c_mask, other=neg_inf).to(tl.float32)
            max_val = tl.maximum(max_val, val)

    out_idx = n * OH_out * OW_out * C + oh_out * OW_out * C + ow_out * C + c_offs
    tl.store(y_ptr + out_idx, max_val.to(tl.float16), mask=c_mask)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self._w = None
        self._cb = None
        self._ab = None
        self._ver = None

    def _cache(self):
        ver = (self.conv.weight._version, self.conv.bias._version, self.bias._version)
        if self._ver != ver:
            w = self.conv.weight
            if w.device.type != "xpu" or w.dtype != torch.float16:
                w = w.to("xpu", dtype=torch.float16)
            self._w = w.permute(2, 3, 1, 0).contiguous()
            b = self.conv.bias
            if b.device.type != "xpu" or b.dtype != torch.float16:
                b = b.to("xpu", dtype=torch.float16)
            self._cb = b.contiguous()
            ab = self.bias.reshape(-1)
            if ab.device.type != "xpu" or ab.dtype != torch.float16:
                ab = ab.to("xpu", dtype=torch.float16)
            self._ab = ab.contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last)
        x_nhwc = x.permute(0, 2, 3, 1)

        N, C_in, H, W = x.shape
        KH, KW, _, C_out = self._w.shape
        OH, OW = H - KH + 1, W - KW + 1

        y_conv = torch.empty((N, C_out, OH, OW), device=x.device,
                             dtype=torch.float16, memory_format=torch.channels_last)
        y_nhwc = y_conv.permute(0, 2, 3, 1)

        grid = lambda meta: (N, OH, triton.cdiv(OW, meta['BLOCK_OW']) * triton.cdiv(C_out, meta['BLOCK_N']))
        _conv2d_tanh_scale_bias_spatial[grid](
            x_nhwc, self._w, self._cb, self._ab, y_nhwc,
            N, H, W, C_out, OH, OW,
            self._w.stride(0), self._w.stride(1), self._w.stride(2), self._w.stride(3),
            float(self.scaling_factor),
            KH=KH, KW=KW, C_IN=C_in,
        )

        # MaxPool via Triton on NHWC
        pool_k = self.max_pool.kernel_size
        if isinstance(pool_k, tuple):
            pool_h, pool_w = pool_k
        else:
            pool_h = pool_w = pool_k
        OH_pool = OH // pool_h
        OW_pool = OW // pool_w

        y_pool_nhwc = torch.empty((N, OH_pool, OW_pool, C_out), device=x.device, dtype=torch.float16)
        BLOCK_C = 64
        num_c_tiles = triton.cdiv(C_out, BLOCK_C)
        grid2 = (N, OH_pool, OW_pool * num_c_tiles)
        _maxpool2d_nhwc_kernel[grid2](
            y_nhwc, y_pool_nhwc,
            N, OH, OW, C_out,
            pool_h, pool_w,
            OH_pool, OW_pool,
            BLOCK_C=BLOCK_C,
        )

        return y_pool_nhwc.permute(0, 3, 1, 2).contiguous()
