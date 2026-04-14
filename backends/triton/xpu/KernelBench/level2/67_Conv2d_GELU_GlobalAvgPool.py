import torch
import torch.nn as nn
import triton
import triton.language as tl


# Conv + GELU + partial row sum (avoids writing full conv output)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W', 'C_IN', 'C_out', 'OH', 'OW'],
)
@triton.jit
def _conv_gelu_rowsum(
    x_ptr, w_ptr, bias_ptr, rowsum_ptr,
    N_batch, H, W, C_out, OH, OW,
    stride_wkh, stride_wkw, stride_wci, stride_wco,
    BLOCK_OW: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, C_IN: tl.constexpr,
):
    """Conv + GELU + sum over ow tile → partial row sums [N, OH, C_out]."""
    n = tl.program_id(0)
    oh = tl.program_id(1)
    pid_ow = tl.program_id(2)
    ow0 = pid_ow * BLOCK_OW
    HW = H * W

    acc = tl.zeros((BLOCK_OW, BLOCK_N), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            x_row = n * HW + (oh + kh) * W + (ow0 + kw)
            x_bp = tl.make_block_ptr(base=x_ptr, shape=(x_row + W - (ow0 + kw), C_IN),
                strides=(C_IN, 1), offsets=(x_row, 0), block_shape=(BLOCK_OW, BLOCK_K), order=(1, 0))
            w_bp = tl.make_block_ptr(base=w_ptr + kh * stride_wkh + kw * stride_wkw,
                shape=(C_IN, C_out), strides=(stride_wci, stride_wco),
                offsets=(0, 0), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
            for c0 in range(0, C_IN, BLOCK_K):
                xt = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                wt = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(xt, wt, acc, input_precision="ieee")
                x_bp = tl.advance(x_bp, (0, BLOCK_K))
                w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    cb = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += cb[None, :]

    # GELU
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.70710678118654752440))

    # Mask out-of-bounds ow positions before summing
    offs_ow = ow0 + tl.arange(0, BLOCK_OW)
    ow_mask = offs_ow < OW
    acc = tl.where(ow_mask[:, None], acc, 0.0)

    # Sum over ow dimension → [BLOCK_N] partial sum for this (n, oh, ow_tile)
    tile_sum = tl.sum(acc, axis=0)  # [BLOCK_N]

    # Write to rowsum[n, oh, ow_tile, c] — no atomic needed, each tile has its own slot
    num_ow_tiles = tl.cdiv(OW, BLOCK_OW)
    base = rowsum_ptr + ((n * OH + oh) * num_ow_tiles + pid_ow) * C_out + offs_n
    tl.store(base, tile_sum.to(tl.float32), mask=mask_n)


# Final reduction: sum across OH rows → [N, C_out], divide by count
@triton.jit
def _reduce_all_kernel(
    rowsum_ptr, y_ptr,
    N_batch, total_slots, C_out, total_count,
    BLOCK_C: tl.constexpr,
):
    """Sum rowsum[n, :, c] across all OH*ow_tiles → y[n, c] / count."""
    n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_out

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for s in range(total_slots):
        vals = tl.load(rowsum_ptr + (n * total_slots + s) * C_out + offs_c, mask=mask_c, other=0.0)
        acc += vals

    out = acc / total_count
    tl.store(y_ptr + n * C_out + offs_c, out.to(tl.float16), mask=mask_c)


def _to(x):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._w = None
        self._ver = None

    def _cache(self):
        ver = (self.conv.weight._version,)
        if self._ver != ver:
            self._w = _to(self.conv.weight).permute(2, 3, 1, 0).contiguous()
            self._b = _to(self.conv.bias).contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        x = _to(x).contiguous(memory_format=torch.channels_last)
        x_nhwc = x.permute(0, 2, 3, 1)

        N, C_in, H, W = x.shape
        KH, KW, _, C_out = self._w.shape
        OH, OW = H - KH + 1, W - KW + 1

        # Partial sums: [N, OH * max_ow_tiles, C_out]
        num_ow_tiles = triton.cdiv(OW, 128)  # BLOCK_OW=128 fixed
        total_slots = OH * num_ow_tiles
        rowsum = torch.empty((N, total_slots, C_out), device=x.device, dtype=torch.float32)

        grid = lambda meta: (N, OH, triton.cdiv(OW, meta['BLOCK_OW']))
        _conv_gelu_rowsum[grid](
            x_nhwc, self._w, self._b, rowsum,
            N, H, W, C_out, OH, OW,
            self._w.stride(0), self._w.stride(1), self._w.stride(2), self._w.stride(3),
            KH=KH, KW=KW, C_IN=C_in,
        )

        # Final reduction: sum all slots → [N, C_out]
        y = torch.empty((N, C_out), device=x.device, dtype=torch.float16)
        _reduce_all_kernel[(N,)](
            rowsum, y, N, total_slots, C_out, float(OH * OW),
            BLOCK_C=64,
        )

        return y  # (N, C_out) — matches reference squeeze
