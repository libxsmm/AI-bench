import math

import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear_affine


class Model(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        bias_shape=(1,),
        divide_value=1.0,
    ):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.register_buffer(
            "_bn_scale",
            torch.tensor(1.0 / math.sqrt(1.0 + bn_eps)),
            persistent=False,
        )
        self._linear_cache = None
        divide_value = hl.constexpr(divide_value)

        def epilogue(x):
            x = x / divide_value
            return x * torch.sigmoid(x)

        self._epilogue = epilogue

    def forward(self, x):
        assert not self.training, "fused BatchNorm epilogue requires eval mode"
        x, self._linear_cache = linear_affine(
            x,
            self.matmul,
            self._bn_scale,
            self.bias,
            self._epilogue,
            self._linear_cache,
        )
        return x
