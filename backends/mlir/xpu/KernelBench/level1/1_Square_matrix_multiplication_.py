import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    mlir_pipeline = "mlp"
    pipeline_parameters = "kb_params_level1-1.json"

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
