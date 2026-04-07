import torch
import torch.nn as nn

import ai_bench.mlir


@torch.compile(
    dynamic=False, backend=ai_bench.mlir.cpu_backend(ai_bench.mlir.cpu_pipeline)
)
class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
