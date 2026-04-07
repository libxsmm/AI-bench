import torch
import torch.nn as nn

import ai_bench.mlir


@torch.compile(
    dynamic=False, backend=ai_bench.mlir.cpu_backend(ai_bench.mlir.cpu_pipeline)
)
class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation,
    group normalization, and then HardSwish activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        eps,
        bias=True,
    ):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = self.group_norm(x)
        x = torch.nn.functional.hardswish(x)  # HardSwish activation
        return x
