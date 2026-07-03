import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- Fused sub + tanh pointwise kernel ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _sub_tanh_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Determine channel index for per-channel bias: NCHW layout
    # element index = n*C*HW + c*HW + hw
    c_idx = (offs // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    y = x - b
    # tanh via sigmoid trick
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0

    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias_shape,
        stride=2,
        padding=1,
        output_padding=1,
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
        self._ct_w = None
        self._ct_b = None
        self._sb = None
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
            sb = self.bias.reshape(-1)
            if sb.device.type != "xpu" or sb.dtype != torch.float16:
                sb = sb.to("xpu", dtype=torch.float16)
            self._sb = sb.contiguous()
            self._ver = ver

    def forward(self, x):
        self._cache()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous()

        # Use vendor conv_transpose2d
        y1 = F.conv_transpose2d(
            x, self._ct_w, self._ct_b, stride=2, padding=1, output_padding=1
        )
        if not y1.is_contiguous():
            y1 = y1.contiguous()

        N, C, H_out, W_out = y1.shape
        y2 = torch.empty_like(y1)
        n_elements = y1.numel()
        HW = H_out * W_out

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _sub_tanh_kernel[grid](
            y1,
            self._sb,
            y2,
            n_elements,
            C,
            HW,
        )
        return y2
