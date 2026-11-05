import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import group_norm
from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self._linear_cache = None
        slope = hl.constexpr(negative_slope)

        def epilogue(y, _):
            # LeakyReLU, then x + x.
            y = torch.where(y >= 0, y, y * slope)
            return y + y

        self._epilogue = epilogue

    def forward(self, x):
        y, self._linear_cache = linear(x, self.fc, cache=self._linear_cache)
        gn = self.gn
        # Not in place: with the output pool it gains nothing, and the kernel's stack
        # copy costs compile time. With the pool off, in place would pay under glibc.
        return group_norm(y, gn.num_groups, gn.weight, gn.bias, gn.eps, self._epilogue)
