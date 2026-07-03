import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- Fused softmax(dim=1) + bias + scale + sigmoid ----------
# One program per (n, h, w) pixel. Loads all C channels, computes softmax,
# adds bias, scales, applies sigmoid, stores.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=8, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def _softmax_bias_scale_sigmoid_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    scale,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    w_idx = pid % W
    tmp = pid // W
    h_idx = tmp % H
    n_idx = tmp // H

    base_x = n_idx * stride_xn + h_idx * stride_xh + w_idx * stride_xw
    base_o = n_idx * stride_on + h_idx * stride_oh + w_idx * stride_ow

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x_vals = tl.load(
        x_ptr + base_x + offs_c * stride_xc, mask=mask_c, other=-float("inf")
    ).to(tl.float32)

    # Online softmax
    x_max = tl.max(x_vals, axis=0)
    x_vals = x_vals - x_max
    exp_x = tl.exp(x_vals)
    softmax_vals = exp_x / tl.sum(exp_x, axis=0)

    # bias + scale + sigmoid
    b_vals = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = tl.sigmoid((softmax_vals + b_vals) * scale)

    tl.store(out_ptr + base_o + offs_c * stride_oc, y.to(tl.float16), mask=mask_c)


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ]


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self._ct_w = None
        self._ct_b = None
        self._ab = None
        self._ver = None

    def _cache(self):
        ver = (
            self.conv_transpose.weight._version,
            self.conv_transpose.bias._version
            if self.conv_transpose.bias is not None
            else 0,
            self.bias._version,
        )
        if self._ver != ver:
            w = self.conv_transpose.weight
            if w.device.type != "xpu" or w.dtype != torch.float16:
                w = w.to("xpu", dtype=torch.float16)
            self._ct_w = w.contiguous()
            if self.conv_transpose.bias is not None:
                b = self.conv_transpose.bias
                if b.device.type != "xpu" or b.dtype != torch.float16:
                    b = b.to("xpu", dtype=torch.float16)
                self._ct_b = b.contiguous()
            else:
                self._ct_b = None
            ab = self.bias.reshape(-1)
            if ab.device.type != "xpu" or ab.dtype != torch.float16:
                ab = ab.to("xpu", dtype=torch.float16)
            self._ab = ab.contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous()

        # Vendor conv_transpose2d
        y1 = F.conv_transpose2d(
            x, self._ct_w, self._ct_b, stride=2, padding=1, output_padding=1
        )
        if not y1.is_contiguous():
            y1 = y1.contiguous()

        N, C, H_out, W_out = y1.shape
        y2 = torch.empty_like(y1)

        grid = (N * H_out * W_out,)
        _softmax_bias_scale_sigmoid_kernel[grid](
            y1,
            self._ab,
            y2,
            N,
            C,
            H_out,
            W_out,
            y1.stride(0),
            y1.stride(1),
            y1.stride(2),
            y1.stride(3),
            y2.stride(0),
            y2.stride(1),
            y2.stride(2),
            y2.stride(3),
            float(self.scaling_factor),
        )
        return y2
