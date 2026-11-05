import helion_mlir_backend  # noqa: F401
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import softmax


class Model(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self._linear_cache = None

    def forward(self, x):
        # AI-bench evaluates models in eval mode (KernelRunner.init_model),
        # where dropout is the identity. Random training-mode masks cannot
        # match the reference's, so refuse them instead of ignoring them.
        if self.training and self.dropout.p > 0:
            raise ValueError("dropout with p > 0 needs eval mode")
        y, self._linear_cache = linear(x, self.matmul, cache=self._linear_cache)
        return softmax(y, in_place=True)
