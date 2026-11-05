import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import group_norm
from helion_mlir_cpu_utils import linear


def _hardtanh_mish(x):
    x = torch.clamp(x, -1.0, 1.0)
    return x * torch.tanh(torch.log1p(torch.exp(x)))


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(
            x,
            self.gemm,
            _hardtanh_mish,
            self._linear_cache,
            biases=(self.gemm.bias, self.bias),
        )
        gn = self.groupnorm
        # Not in place: with the output pool it gains nothing, and the kernel's stack
        # copy costs compile time. With the pool off, in place would pay under glibc.
        return group_norm(y, gn.num_groups, gn.weight, gn.bias, gn.eps)
