import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_OW": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_OW": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_OW": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
    ],
    key=["H", "W", "C_IN", "C_out", "OH", "OW"],
)
@triton.jit
def _conv2d_relu_bias_spatial(
    x_ptr,
    w_ptr,
    conv_bias_ptr,
    bias_ptr,
    y_ptr,
    N_batch,
    H,
    W,
    C_out,
    OH,
    OW,
    stride_wkh,
    stride_wkw,
    stride_wci,
    stride_wco,
    BLOCK_OW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    C_IN: tl.constexpr,
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
            x_row_start = n * HW + (oh + kh) * W + (ow0 + kw)
            x_valid_rows = W - (ow0 + kw)
            x_bp = tl.make_block_ptr(
                base=x_ptr,
                shape=(x_row_start + x_valid_rows, C_IN),
                strides=(C_IN, 1),
                offsets=(x_row_start, 0),
                block_shape=(BLOCK_OW, BLOCK_K),
                order=(1, 0),
            )
            w_bp = tl.make_block_ptr(
                base=w_ptr + kh * stride_wkh + kw * stride_wkw,
                shape=(C_IN, C_out),
                strides=(stride_wci, stride_wco),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
            for c0 in range(0, C_IN, BLOCK_K):
                x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                w_tile = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
                x_bp = tl.advance(x_bp, (0, BLOCK_K))
                w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    conv_b = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    ext_b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += conv_b[None, :]
    acc = tl.maximum(acc, 0.0)
    acc += ext_b[None, :]

    OHOW = OH * OW
    y_row_start = n * OHOW + oh * OW + ow0
    y_valid_rows = OW - ow0
    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(y_row_start + y_valid_rows, C_out),
        strides=(C_out, 1),
        offsets=(y_row_start, pid_n * BLOCK_N),
        block_shape=(BLOCK_OW, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


def _ensure_xpu_fp16(x):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


def kernel_function(x, w_hwio, conv_bias, post_bias):
    x_xpu = _ensure_xpu_fp16(x).contiguous(memory_format=torch.channels_last)
    x_nhwc = x_xpu.permute(0, 2, 3, 1)
    N, C_in, H, W = x_xpu.shape
    KH, KW, _, C_out = w_hwio.shape
    OH, OW = H - KH + 1, W - KW + 1
    y = torch.empty(
        (N, C_out, OH, OW),
        device=x_xpu.device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    y_nhwc = y.permute(0, 2, 3, 1)
    cb = conv_bias.view(-1)
    pb = post_bias.view(-1)

    def grid(meta):
        return (
            N,
            OH,
            triton.cdiv(OW, meta["BLOCK_OW"]) * triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    _conv2d_relu_bias_spatial[grid](
        x_nhwc,
        w_hwio,
        cb,
        pb,
        y_nhwc,
        N,
        H,
        W,
        C_out,
        OH,
        OW,
        w_hwio.stride(0),
        w_hwio.stride(1),
        w_hwio.stride(2),
        w_hwio.stride(3),
        KH=KH,
        KW=KW,
        C_IN=C_in,
    )
    return y


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._w_hwio = None
        self._w_sig = None

    def forward(self, x):
        sig = (self.conv.weight.data_ptr(), self.conv.weight._version)
        if self._w_hwio is None or self._w_sig != sig:
            self._w_hwio = (
                _ensure_xpu_fp16(self.conv.weight).permute(2, 3, 1, 0).contiguous()
            )
            self._w_sig = sig
        cb = _ensure_xpu_fp16(self.conv.bias).contiguous()
        pb = _ensure_xpu_fp16(self.bias).contiguous()
        return kernel_function(x, self._w_hwio, cb, pb)
