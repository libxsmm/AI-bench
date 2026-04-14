import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 7: Conv3d(8->32, k=3, no padding) + ReLU + LeakyReLU + GELU + Sigmoid + BiasAdd
#
# Single fused spatial-tiled Conv3d kernel.
# Note: relu then leaky_relu(0.01) on already-relu'd values = just relu.
# Epilogue: conv_bias + relu + gelu + sigmoid + bias_add (per-channel)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['D', 'H', 'W', 'C_IN', 'C_OUT', 'OD', 'OH', 'OW'],
)
@triton.jit
def _conv3d_relu_gelu_sigmoid_biasadd_kernel(
    x_ptr, w_ptr, b_ptr, add_bias_ptr, y_ptr,
    N_batch, D, H, W, OD, OH, OW,
    sx_n, sx_d, sx_h,
    sw_kd, sw_kh, sw_kw, sw_ci, sw_co,
    sy_n, sy_d, sy_h,
    BLOCK_OW: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
):
    n = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    od = pid_dh // OH
    oh = pid_dh % OH
    ow0 = pid_ow * BLOCK_OW

    acc = tl.zeros((BLOCK_OW, BLOCK_N), dtype=tl.float32)
    x_n_base = x_ptr + n * sx_n

    for kd in range(KD):
        for kh in range(KH):
            x_dh_base = x_n_base + (od + kd) * sx_d + (oh + kh) * sx_h
            for kw in range(KW):
                w_start = ow0 + kw
                x_bp = tl.make_block_ptr(
                    base=x_dh_base, shape=(W, C_IN),
                    strides=(C_IN, 1), offsets=(w_start, 0),
                    block_shape=(BLOCK_OW, BLOCK_K), order=(1, 0),
                )
                w_bp = tl.make_block_ptr(
                    base=w_ptr + kd * sw_kd + kh * sw_kh + kw * sw_kw,
                    shape=(C_IN, C_OUT), strides=(sw_ci, sw_co),
                    offsets=(0, 0), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
                )
                for c0 in range(0, C_IN, BLOCK_K):
                    xt = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                    wt = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                    acc = tl.dot(xt, wt, acc, input_precision="ieee")
                    x_bp = tl.advance(x_bp, (0, BLOCK_K))
                    w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    # Epilogue: conv_bias + relu + leaky_relu(0.01) [= relu] + gelu + sigmoid + bias_add
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_OUT
    conv_bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += conv_bias[None, :]

    # ReLU (relu then leaky_relu(0.01) on relu output = relu)
    acc = tl.maximum(acc, 0.0)

    # LeakyReLU(0.01) on relu'd values -- values are already >= 0, so this is identity
    # but for correctness we include it:
    acc = tl.where(acc >= 0.0, acc, acc * 0.01)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.70710678118654752440))

    # Sigmoid
    acc = tl.sigmoid(acc)

    # BiasAdd (per-channel)
    add_b = tl.load(add_bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += add_b[None, :]

    # Store
    y_dh_base = y_ptr + n * sy_n + od * sy_d + oh * sy_h
    y_valid = OW - ow0
    y_bp = tl.make_block_ptr(
        base=y_dh_base, shape=(y_valid, C_OUT),
        strides=(C_OUT, 1), offsets=(0, 0),
        block_shape=(BLOCK_OW, BLOCK_N), order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


def _to_xpu_fp16(x):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._w = None
        self._ver = None

    def _cache(self):
        ver = (self.conv.weight._version,)
        if self._ver != ver:
            # Weight: (C_out, C_in, KD, KH, KW) -> (KD, KH, KW, C_in, C_out)
            self._w = _to_xpu_fp16(self.conv.weight).permute(2, 3, 4, 1, 0).contiguous()
            self._b = _to_xpu_fp16(self.conv.bias).contiguous()
            self._add_bias = _to_xpu_fp16(self.bias).view(-1).contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        x = _to_xpu_fp16(x).contiguous(memory_format=torch.channels_last_3d)
        x_ndhwc = x.permute(0, 2, 3, 4, 1)

        N, C_in, D_x, H_x, W_x = x.shape
        KD, KH, KW, _, C_out = self._w.shape
        OD = D_x - KD + 1
        OH = H_x - KH + 1
        OW = W_x - KW + 1

        y = torch.empty((N, C_out, OD, OH, OW), device=x.device,
                         dtype=torch.float16, memory_format=torch.channels_last_3d)
        y_ndhwc = y.permute(0, 2, 3, 4, 1)

        grid = lambda meta: (N, OD * OH, triton.cdiv(OW, meta['BLOCK_OW']))

        _conv3d_relu_gelu_sigmoid_biasadd_kernel[grid](
            x_ndhwc, self._w, self._b, self._add_bias, y_ndhwc,
            N, D_x, H_x, W_x, OD, OH, OW,
            x_ndhwc.stride(0), x_ndhwc.stride(1), x_ndhwc.stride(2),
            self._w.stride(0), self._w.stride(1), self._w.stride(2), self._w.stride(3), self._w.stride(4),
            y_ndhwc.stride(0), y_ndhwc.stride(1), y_ndhwc.stride(2),
            KD=KD, KH=KH, KW=KW, C_IN=C_in, C_OUT=C_out,
        )
        return y
