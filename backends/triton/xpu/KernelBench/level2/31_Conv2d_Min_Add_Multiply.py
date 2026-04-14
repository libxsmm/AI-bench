import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W', 'C_IN', 'C_out', 'OH', 'OW'],
)
@triton.jit
def _fused_conv_spatial(
    x_ptr, w_ptr, conv_bias_ptr, post_bias_ptr, y_ptr,
    N_batch, H, W, C_out, OH, OW,
    stride_wkh, stride_wkw, stride_wci, stride_wco,
    constant_value, scaling_factor,
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
                xt = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                wt = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(xt, wt, acc, input_precision="ieee")
                x_bp = tl.advance(x_bp, (0, BLOCK_K))
                w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    # Epilogue: + conv_bias → min(constant) → + post_bias → * scaling
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    cb = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    pb = tl.load(post_bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += cb[None, :]
    acc = tl.minimum(acc, constant_value)
    acc += pb[None, :]
    acc *= scaling_factor

    y_row = n * OH * OW + oh * OW + ow0
    y_bp = tl.make_block_ptr(
        base=y_ptr, shape=(y_row + OW - ow0, C_out),
        strides=(C_out, 1), offsets=(y_row, pid_n * BLOCK_N),
        block_shape=(BLOCK_OW, BLOCK_N), order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


def _to_xpu_fp16(x):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self._w = None
        self._ver = None

    def _cache(self):
        ver = (self.conv.weight._version, self.conv.bias._version, self.bias._version)
        if self._ver != ver:
            self._w = _to_xpu_fp16(self.conv.weight).permute(2, 3, 1, 0).contiguous()
            self._cb = _to_xpu_fp16(self.conv.bias).contiguous()
            self._pb = _to_xpu_fp16(self.bias).view(-1).contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        x = _to_xpu_fp16(x).contiguous(memory_format=torch.channels_last)
        x_nhwc = x.permute(0, 2, 3, 1)
        N, C_in, H, W = x.shape
        KH, KW, _, C_out = self._w.shape
        OH, OW = H - KH + 1, W - KW + 1
        y = torch.empty((N, C_out, OH, OW), device=x.device,
                         dtype=torch.float16, memory_format=torch.channels_last)
        y_nhwc = y.permute(0, 2, 3, 1)

        grid = lambda meta: (N, OH, triton.cdiv(OW, meta['BLOCK_OW']) * triton.cdiv(C_out, meta['BLOCK_N']))
        _fused_conv_spatial[grid](
            x_nhwc, self._w, self._cb, self._pb, y_nhwc,
            N, H, W, C_out, OH, OW,
            self._w.stride(0), self._w.stride(1), self._w.stride(2), self._w.stride(3),
            float(self.constant_value), float(self.scaling_factor),
            KH=KH, KW=KW, C_IN=C_in,
        )
        return y
