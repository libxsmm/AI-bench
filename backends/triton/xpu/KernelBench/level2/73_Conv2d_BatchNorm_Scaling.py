import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Spatial-tiled Conv2d + bias (NHWC layout, block_ptr) ----------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_OW": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8, num_stages=2
        ),
    ],
    key=["H", "W", "C_IN", "C_out", "OH", "OW"],
)
@triton.jit
def _conv2d_bn_scale_spatial(
    x_ptr,
    w_ptr,
    conv_bias_ptr,
    bn_scale_ptr,
    bn_shift_ptr,
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
            x_row = n * HW + (oh + kh) * W + (ow0 + kw)
            x_bp = tl.make_block_ptr(
                base=x_ptr,
                shape=(x_row + W - (ow0 + kw), C_IN),
                strides=(C_IN, 1),
                offsets=(x_row, 0),
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

    # Fused: conv_bias + BN(eval) + scaling all in one
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    cb = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += cb[None, :]
    # BN+scale: (x - mean) / sqrt(var+eps) * weight * scaling_factor + bias * scaling_factor
    # Precomputed as: x * bn_scale + bn_shift (includes scaling_factor)
    bn_s = tl.load(bn_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    bn_b = tl.load(bn_shift_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc * bn_s[None, :] + bn_b[None, :]

    # store
    OHOW = OH * OW
    y_row = n * OHOW + oh * OW + ow0
    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(y_row + OW - ow0, C_out),
        strides=(C_out, 1),
        offsets=(y_row, pid_n * BLOCK_N),
        block_shape=(BLOCK_OW, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# ---------- Fused BatchNorm + Scaling pointwise kernel (NHWC layout) ----------
@triton.jit
def _batchnorm_scale_nhwc_kernel(
    x_ptr,
    y_ptr,
    bn_scale_ptr,
    bn_shift_ptr,
    total_hw,
    C,
    BLOCK_C: tl.constexpr,
):
    # Grid: (total_hw,) where total_hw = N * OH * OW
    # BN scale/shift already include the scaling_factor
    pid = tl.program_id(0)

    for c0 in range(0, C, BLOCK_C):
        c_offs = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        scale = tl.load(bn_scale_ptr + c_offs, mask=c_mask, other=1.0).to(tl.float32)
        shift = tl.load(bn_shift_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        idx = pid * C + c_offs
        val = tl.load(x_ptr + idx, mask=c_mask, other=0.0).to(tl.float32)
        out = val * scale + shift
        tl.store(y_ptr + idx, out.to(tl.float16), mask=c_mask)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self._w = None
        self._cb = None
        self._bn_scale = None
        self._bn_shift = None
        self._ver = None

    def _cache(self):
        ver = (self.conv.weight._version, self.conv.bias._version)
        if self._ver != ver:
            w = self.conv.weight
            if w.device.type != "xpu" or w.dtype != torch.float16:
                w = w.to("xpu", dtype=torch.float16)
            self._w = w.permute(2, 3, 1, 0).contiguous()
            b = self.conv.bias
            if b.device.type != "xpu" or b.dtype != torch.float16:
                b = b.to("xpu", dtype=torch.float16)
            self._cb = b.contiguous()
            self._ver = ver

    def _cache_bn(self):
        # Precompute fused BN + scaling: x * (bn_scale * sf) + (bn_shift * sf)
        bn_w = self.bn.weight.float()
        bn_b = self.bn.bias.float()
        rm = self.bn.running_mean.float()
        rv = self.bn.running_var.float()
        eps = self.bn.eps
        sf = float(self.scaling_factor)
        bn_scale = bn_w / torch.sqrt(rv + eps)
        bn_shift = bn_b - rm * bn_scale
        # Fuse scaling_factor into BN params
        self._bn_scale = (bn_scale * sf).to("xpu", dtype=torch.float16).contiguous()
        self._bn_shift = (bn_shift * sf).to("xpu", dtype=torch.float16).contiguous()

    def forward(self, x):
        self._cache()
        self._cache_bn()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last)
        x_nhwc = x.permute(0, 2, 3, 1)

        N, C_in, H, W = x.shape
        KH, KW, _, C_out = self._w.shape
        OH, OW = H - KH + 1, W - KW + 1

        y_conv = torch.empty(
            (N, C_out, OH, OW),
            device=x.device,
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        y_nhwc = y_conv.permute(0, 2, 3, 1)

        grid = lambda meta: (
            N,
            OH,
            triton.cdiv(OW, meta["BLOCK_OW"]) * triton.cdiv(C_out, meta["BLOCK_N"]),
        )
        _conv2d_bn_scale_spatial[grid](
            x_nhwc,
            self._w,
            self._cb,
            self._bn_scale,
            self._bn_shift,
            y_nhwc,
            N,
            H,
            W,
            C_out,
            OH,
            OW,
            self._w.stride(0),
            self._w.stride(1),
            self._w.stride(2),
            self._w.stride(3),
            KH=KH,
            KW=KW,
            C_IN=C_in,
        )

        return y_conv
