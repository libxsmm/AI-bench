import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import group_norm
from helion_mlir_cpu_utils import linear


def _epilogue(y, multiply_weight):
    y = y * torch.sigmoid(y)
    y = y * multiply_weight
    return y * torch.sigmoid(y)


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        gn = self.group_norm
        return group_norm(
            y,
            gn.num_groups,
            gn.weight,
            gn.bias,
            gn.eps,
            _epilogue,
            self.multiply_weight,
            in_place=True,
        )
