import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    mlir_pipeline = "mlp"
    pipeline_parameters = "kb_params_level1-17.json"

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B.T)
