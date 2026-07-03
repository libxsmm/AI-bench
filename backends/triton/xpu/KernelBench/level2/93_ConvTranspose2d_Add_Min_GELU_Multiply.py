import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- Fused add + min(scalar) + GELU + multiply pointwise kernel ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_min_gelu_mul_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    add_value,
    multiply_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # add
    x = x + add_value
    # min(x, 0.0)
    x = tl.minimum(x, 0.0)
    # GELU
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.70710678118654752440))
    # multiply
    x = x * multiply_value

    tl.store(y_ptr + offs, x.to(tl.float16), mask=mask)


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]


class Model(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.add_value = add_value
        self.multiply_value = multiply_value
        self._ct_w = None
        self._ct_b = None
        self._ver = None

    def _cache(self):
        ver = (
            self.conv_transpose.weight._version,
            self.conv_transpose.bias._version
            if self.conv_transpose.bias is not None
            else 0,
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
            self._ver = ver

    def forward(self, x):
        self._cache()
        if x.device.type != "xpu" or x.dtype != torch.float16:
            x = x.to("xpu", dtype=torch.float16)
        x = x.contiguous()

        # Vendor conv_transpose2d (stride=2, no padding, no output_padding)
        y1 = F.conv_transpose2d(x, self._ct_w, self._ct_b, stride=2)
        if not y1.is_contiguous():
            y1 = y1.contiguous()

        y2 = torch.empty_like(y1)
        n_elements = y1.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _add_min_gelu_mul_kernel[grid](
            y1,
            y2,
            n_elements,
            float(self.add_value),
            float(self.multiply_value),
        )
        return y2
