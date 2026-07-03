import torch
import torch.nn as nn
import triton
import triton.language as tl


# Spatial-tiled Conv (ConvTranspose2d stride=1 = Conv2d with flipped weight)
@triton.autotune(
    configs=[
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
def _conv_spatial(
    x_ptr,
    w_ptr,
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
    pid_ow = tl.program_id(2)
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
                offsets=(0, 0),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
            for c0 in range(0, C_IN, BLOCK_K):
                xt = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                wt = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(xt, wt, acc, input_precision="ieee")
                x_bp = tl.advance(x_bp, (0, BLOCK_K))
                w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    OHOW = OH * OW
    y_row = n * OHOW + oh * OW + ow0
    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(y_row + OW - ow0, C_out),
        strides=(C_out, 1),
        offsets=(y_row, 0),
        block_shape=(BLOCK_OW, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# Fused MaxPool(2x2) + Hardtanh + Mean(dim=2,3) + Tanh
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 64, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 128, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 64, "BLOCK_C": 32}, num_warps=4, num_stages=2),
    ],
    key=["C", "H_pool", "W_pool"],
)
@triton.jit
def _fused_pool_hardtanh_mean_tanh(
    conv_ptr,
    y_ptr,
    N,
    C,
    H_conv,
    W_conv,
    H_pool,
    W_pool,
    sc_n,
    sc_h,
    sc_w,
    sc_c,
    hardtanh_min,
    hardtanh_max,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused: MaxPool(2x2) → Hardtanh → partial sum for Mean → final Tanh."""
    n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Accumulate sum over all pooled spatial positions for Mean
    running_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    count = H_pool * W_pool

    for h_pool in range(H_pool):
        for w_tile in range(0, W_pool, BLOCK_W):
            offs_w = w_tile + tl.arange(0, BLOCK_W)
            mask_w = offs_w < W_pool

            # MaxPool over 2x2
            pooled = tl.full((BLOCK_C, BLOCK_W), -float("inf"), dtype=tl.float32)
            for hh in range(2):
                h_in = h_pool * 2 + hh
                for ww in range(2):
                    w_in = offs_w * 2 + ww
                    ptrs = (
                        conv_ptr
                        + n * sc_n
                        + h_in * sc_h
                        + w_in[None, :] * sc_w
                        + offs_c[:, None] * sc_c
                    )
                    vals = tl.load(
                        ptrs,
                        mask=mask_c[:, None] & mask_w[None, :],
                        other=-float("inf"),
                    ).to(tl.float32)
                    pooled = tl.maximum(pooled, vals)

            # Hardtanh
            pooled = tl.minimum(tl.maximum(pooled, hardtanh_min), hardtanh_max)

            # Accumulate for mean (masked)
            masked_pooled = tl.where(mask_w[None, :], pooled, 0.0)
            running_sum += tl.sum(masked_pooled, axis=1)

    # Mean + Tanh
    mean_val = running_sum / count
    # tanh = (exp(2x) - 1) / (exp(2x) + 1)
    exp2x = tl.exp(2.0 * mean_val)
    out = (exp2x - 1.0) / (exp2x + 1.0)

    # Store: y shape (N, C, 1, 1) — just write per (n, c)
    tl.store(y_ptr + n * C + offs_c, out.to(tl.float16), mask=mask_c)


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ]


def _to_xpu_fp16(x):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_stride
        )
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self._w = None
        self._ver = None

    def _cache(self):
        ver = (self.conv_transpose.weight._version, self.conv_transpose.bias._version)
        if self._ver != ver:
            # ConvTranspose2d stride=1 = Conv2d with w.transpose(0,1).flip(2,3)
            w = _to_xpu_fp16(self.conv_transpose.weight)
            w_conv = w.transpose(0, 1).flip(2, 3)
            self._w = w_conv.permute(2, 3, 1, 0).contiguous()  # HWIO
            self._b = _to_xpu_fp16(self.conv_transpose.bias).contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        x = _to_xpu_fp16(x).contiguous(memory_format=torch.channels_last)
        x_nhwc = x.permute(0, 2, 3, 1)

        N, C_in, H, W = x.shape
        KH, KW, _, C_out = self._w.shape
        OH, OW = H - KH + 1, W - KW + 1

        # Conv output in channels_last
        conv_out = torch.empty(
            (N, C_out, OH, OW),
            device=x.device,
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        conv_nhwc = conv_out.permute(0, 2, 3, 1)

        grid = lambda meta: (N, OH, triton.cdiv(OW, meta["BLOCK_OW"]))
        _conv_spatial[grid](
            x_nhwc,
            self._w,
            self._b,
            conv_nhwc,
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

        # Fused MaxPool + Hardtanh + Mean + Tanh
        H_pool = OH // 2
        W_pool = OW // 2
        y = torch.empty((N, C_out, 1, 1), device=x.device, dtype=torch.float16)
        # y is contiguous NCHW with H=W=1, so it's just (N, C) flat
        y_flat = y.view(N, C_out)

        sc = conv_out.stride()
        pool_grid = lambda meta: (N, triton.cdiv(C_out, meta["BLOCK_C"]))

        _fused_pool_hardtanh_mean_tanh[pool_grid](
            conv_out,
            y_flat,
            N,
            C_out,
            OH,
            OW,
            H_pool,
            W_pool,
            sc[0],
            sc[2],
            sc[3],
            sc[1],  # n, h, w, c strides
            float(self.hardtanh_min),
            float(self.hardtanh_max),
        )
        return y
