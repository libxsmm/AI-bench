import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import softmax


def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.linear, _gelu, self._linear_cache)
        return softmax(y, in_place=True)
