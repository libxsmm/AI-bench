import math

import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear_affine


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._linear_cache = None
        bn_scale = hl.constexpr(1.0 / math.sqrt(1.0 + eps))

        def epilogue(x):
            return x * bn_scale

        self._epilogue = epilogue

    def forward(self, x):
        assert not self.training, "fused BatchNorm epilogue requires eval mode"
        x, self._linear_cache = linear_affine(
            x,
            self.gemm,
            self.scale,
            epilogue=self._epilogue,
            cache=self._linear_cache,
        )
        return x
